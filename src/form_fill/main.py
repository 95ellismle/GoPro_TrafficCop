from playwright.sync_api import BrowserContext, sync_playwright

from undetected_playwright import stealth_sync

headless = True



URL = "https://www.met.police.uk/ro/report/rti/rti-beta-2.1/report-a-road-traffic-incident/"


class FormFill:
    def __init__(self, url):
        self.url = url

    def __enter__(self):
        with sync_playwright() as self.playwright:
            self.browser = self.playwright.chromium.launch(headless=False)
            self.context = self.browser.new_context()
            stealth_sync(self.context)
            self.page = self.context.new_page()
            self.page.goto(URL)

            self.page.screenshot(path='/Users/mattellis/tmp.png', full_page=True)
            import ipdb; ipdb.set_trace()
            return self

    def __exit__(self, *args, **kwargs):
        print(args, kwargs)

    def add_location(self, lat, lon):
        """Add the location to the form"""
        import ipdb; ipdb.set_trace()

