from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(parser_threads=4, downloader_threads=4, storage={'root_dir': '上戸彩'})
crawler.crawl(keyword='上戸彩', max_num=100)