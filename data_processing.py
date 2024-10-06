import pandas as pd

data_books = pd.read_json('cleaned_books.json', lines=True)

books_cleaned = []
for index, book in data_books.iterrows():
    id_book = book['id']
    author = book['author']
    title = book['title']
    summary = book['summary'].replace('\n', ' ').strip()

    books_cleaned.append({
        'id': id_book,
        'author': author,
        'title': title,
        'summary': summary
    })

cleaned_books_df = pd.DataFrame(books_cleaned)
cleaned_books_df.to_json('cleaned_books.json', orient='records', lines=True)
