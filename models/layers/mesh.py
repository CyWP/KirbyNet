from tempfile import mkstemp
from shutil import move
from scipy.sparse import lil_matrix, csr_matrix
import torch
import numpy as np
import os
from models.layers.mesh_union import MeshUnion
from models.layers.mesh_prepare import fill_mesh
from models.layers.mesh_reconstruction import MeshReconstruction


class Mesh:

    def __init__(self, file=None, opt=None, hold_history=True, export_folder="", max_num_edges=None):
        """
        Notes for self on member variables
        vs: (V x 3 list)vertices with actual 3d coordinates
        ve: (V x ~6 list)holds edge_id (idx in edges) of every edge connected to a given vertex. Same size as
        edges: holds idx to two points in vs
        edge_areas:
        v_mask: mask for vertices that still "exist": corresponding to idxs in vs, True if exists, False if not
        pool_count:
        features:
        edge_areas:
        history_data:
        """
        self.vs = self.v_mask = self.filename = self.features = self.edge_areas = None
        self.origin_file =file
        self.edges = self.gemm_edges = self.sides = None
        self.pool_count = 0
        fill_mesh(self, file, opt)
        self.export_folder = export_folder
        self.history_data = None
        self.reconstruction = MeshReconstruction(self)
        if hold_history:
            self.init_history()
        self.export()

    def extract_features(self):
        return self.features

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.v_mask[edge[1]] = False  # masking point since it no longer is used
        mask = self.edges == edge[1]  # Gets all indices at which point was used
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[
            0
        ]  # Essentially replacing all connection of since delted point with the new averaged point.
        #Update face representations
        iv_a = edge[0]
        iv_b = edge[1]
        for f_idx in self.vf[iv_b]:
            f = self.faces[f_idx]
            if f_idx in self.vf[iv_a]:
                self.vf[iv_a].remove(f_idx)
                self.faces_mask[f_idx] = False
            else:
                self.faces[f_idx][f.index(iv_b)] = iv_a
        self.vf[iv_b] = []

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        #only edits self.ve
        vs = self.edges[edge_id]
        for v in vs:
            if edge_id not in self.ve[v]:
                print(self.ve[v])
                print(self.filename)
            self.ve[v].remove(edge_id)

    def clean(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            # if self.v_mask[v_index]:
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve
        self.__clean_history(groups, torch_mask)
        self.pool_count += 1
        self.export()

    def export(self, file=None, vcolor=None):
        if file is None:
            if self.export_folder:
                filename, file_extension = os.path.splitext(self.filename)
                file = "%s/%s_%d%s" % (
                    self.export_folder,
                    filename,
                    self.pool_count,
                    file_extension,
                )
            else:
                return
        faces = []
        vs = self.vs[self.v_mask]
        gemm = np.array(self.gemm_edges)
        new_indices = np.zeros(self.v_mask.shape[0], dtype=np.int32)
        new_indices[self.v_mask] = np.arange(0, np.ma.where(self.v_mask)[0].shape[0])
        for edge_index in range(len(gemm)):
            cycles = self.__get_cycle(gemm, edge_index)
            for cycle in cycles:
                faces.append(self.__cycle_to_face(cycle, new_indices))
        with open(file, "w+") as f:
            for vi, v in enumerate(vs):
                vcol = (
                    " %f %f %f" % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2])
                    if vcolor is not None
                    else ""
                )
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(faces) - 1):
                f.write(
                    "f %d %d %d\n"
                    % (
                        faces[face_id][0] + 1,
                        faces[face_id][1] + 1,
                        faces[face_id][2] + 1,
                    )
                )
            f.write(
                "f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1)
            )
            for edge in self.edges:
                f.write(
                    "\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1)
                )

    def export_segments(self, segments):
        if not self.export_folder:
            return
        cur_segments = segments
        for i in range(self.pool_count + 1):
            filename, file_extension = os.path.splitext(self.filename)
            file = "%s/%s_%d%s" % (self.export_folder, filename, i, file_extension)
            fh, abs_path = mkstemp()
            edge_key = 0
            with os.fdopen(fh, "w") as new_file:
                with open(file) as old_file:
                    for line in old_file:
                        if line[0] == "e":
                            new_file.write(
                                "%s %d" % (line.strip(), cur_segments[edge_key])
                            )
                            if edge_key < len(cur_segments):
                                edge_key += 1
                                new_file.write("\n")
                        else:
                            new_file.write(line)
            os.remove(file)
            move(abs_path, file)
            if i < len(self.history_data["edges_mask"]):
                cur_segments = segments[: len(self.history_data["edges_mask"][i])]
                cur_segments = cur_segments[self.history_data["edges_mask"][i]]

    def __get_cycle(self, gemm, edge_id):
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    def __cycle_to_face(self, cycle, v_indices):
        face = []
        for i in range(3):
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face

    def init_history(self):
        self.history_data = {
            "groups": [],
            "gemm_edges": [self.gemm_edges.copy()],
            "occurrences": [],
            "old2current": np.arange(self.edges_count, dtype=np.int32),
            "current2old": np.arange(self.edges_count, dtype=np.int32),
            "edges_mask": [torch.ones(self.edges_count, dtype=torch.bool)],
            "edges_count": [self.edges_count],
        }
        if self.export_folder:
            self.history_data["collapses"] = MeshUnion(self.edges_count)

    def union_groups(self, source, target):
        if self.export_folder and self.history_data:
            self.history_data["collapses"].union(
                self.history_data["current2old"][source],
                self.history_data["current2old"][target],
            )
        return

    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data["edges_mask"][-1][
                self.history_data["current2old"][index]
            ] = 0
            self.history_data["old2current"][
                self.history_data["current2old"][index]
            ] = -1
            if self.export_folder:
                self.history_data["collapses"].remove_group(
                    self.history_data["current2old"][index]
                )

    def get_groups(self):
        return self.history_data["groups"].pop()

    def get_occurrences(self):
        return self.history_data["occurrences"].pop()

    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            mask = self.history_data["old2current"] != -1
            self.history_data["old2current"][mask] = np.arange(
                self.edges_count, dtype=np.int32
            )
            self.history_data["current2old"][0 : self.edges_count] = np.ma.where(mask)[
                0
            ]
            if self.export_folder != "":
                self.history_data["edges_mask"].append(
                    self.history_data["edges_mask"][-1].clone()
                )
            self.history_data["occurrences"].append(groups.get_occurrences())
            self.history_data["groups"].append(groups.get_groups(pool_mask))
            self.history_data["gemm_edges"].append(self.gemm_edges.copy())
            self.history_data["edges_count"].append(self.edges_count)

    def unroll_gemm(self):
        self.history_data["gemm_edges"].pop()
        self.gemm_edges = self.history_data["gemm_edges"][-1]
        self.history_data["edges_count"].pop()
        self.edges_count = self.history_data["edges_count"][-1]

    def get_edge_areas(self):
        return self.edge_areas

    def mean_curvature_energy(self):
        V = self.vs[self.v_mask, :]
        L = csr_matrix(self.compute_laplacian())
        return np.mean(np.abs(L @ V))

    def compute_laplacian(self):
        faces = self.faces[self.faces_mask]
        vertices = self.vs
        n_vertices = vertices.shape[0]
        # Initialize a sparse matrix for the Laplacian
        L = lil_matrix((n_vertices, n_vertices))
        # Loop over all triangles (faces)
        for face in faces:
            i, j, k = face
            # Vertices of the triangle
            vi, vj, vk = vertices[i], vertices[j], vertices[k]
            # Cotangent weights for the edges
            cot_jk = 0.5 * cotangent(vi, vj, vk)  # Cotangent at vertex vi (opposite edge vj-vk)
            cot_ik = 0.5 * cotangent(vj, vk, vi)  # Cotangent at vertex vj (opposite edge vi-vk)
            cot_ij = 0.5 * cotangent(vk, vi, vj)  # Cotangent at vertex vk (opposite edge vi-vj)
            # Update the Laplacian matrix using cotangent weights
            L[i, i] -= cot_ij + cot_ik
            L[j, j] -= cot_jk + cot_ij
            L[k, k] -= cot_jk + cot_ik
            L[i, j] += cot_ij
            L[j, i] += cot_ij
            L[i, k] += cot_ik
            L[k, i] += cot_ik
            L[j, k] += cot_jk
            L[k, j] += cot_jk
        valid_idx = np.where(self.v_mask)[0]
        return L[valid_idx, :][:, valid_idx]

def cotangent(v1, v2, v3):
    # Vectors along the triangle edges
    u = v2 - v1
    v = v3 - v1
    # Compute cotangent using dot product and cross product
    cos_angle = np.dot(u, v)
    sin_angle = np.linalg.norm(np.cross(u, v))
    return cos_angle / (sin_angle + 10e-6)
