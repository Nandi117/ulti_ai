import urllib.request, bs4

urls = [
    'https://ultiblog.hu/ulti/',
    'https://ultiblog.hu/betli-rebetli-teritett-betli/',
    'https://ultiblog.hu/durchmars-redurchmars-teritett-durchmars/',
    'https://ultiblog.hu/40-100-piros-40-100/'
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req).read()
        soup = bs4.BeautifulSoup(html, 'html.parser')
        # Extract main content, usually in paragraphs
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        print(f"\n--- {url} ---")
        print(text[:1000] + "...")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
