import scrapy

class CurvinSpider(scrapy.Spider):
    """Scrapy spider to crawl and chunk blog content."""
    name = "curvin"
    allowed_domains = ["swervincurvin.blogspot.com"]
    start_urls = [
        "https://swervincurvin.blogspot.com/2025/10/",
        "https://swervincurvin.blogspot.com/2025/11/",
    ]

    def parse(self, response):
        # The full parsing logic will be added later
        self.logger.info(f"Crawled archive: {response.url}")
