from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(storage={'root_dir': '浜辺みなみ'})
crawler.crawl(keyword='浜辺みなみ 女優', max_num=50)
