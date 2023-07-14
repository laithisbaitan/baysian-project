from newspaper import Article

url = 'https://www.aljazeera.com/news/2023/7/14/why-do-some-palestinian-teens-in-jenin-dream-of-martyrdom'
art = Article(url)

art.download()
art.parse()

title = art.title
text = art.text
source = art.source_url
authors = art.authors

if authors:
    print('The author(s) of the article are:', authors)
else:
    print('No authors were found for this article.')
