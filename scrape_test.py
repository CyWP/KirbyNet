from playwright.sync_api import sync_playwright

url = "https://www.thingiverse.com/download:32779"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url)
    print("Final URL:", page.url)
    browser.close()
