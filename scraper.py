import requests
from bs4 import BeautifulSoup
import time
import json


BASE_URL = "https://www.gutenberg.org/ebooks/"


# Fonction d'extraction des informations d'un livre en utilisant son ID unique
def get_book_info(book_id):
    url = f"{BASE_URL}{book_id}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extraction de l'auteur
        author = soup.find('th',
                           text='Author').find_next('td').get_text(strip=True)

        # Extraction du titre
        title = soup.find('th',
                          text='Title').find_next('td').get_text(strip=True)

        # Extraction du résumé
        summary = soup.find('th',
                            text='Summary').find_next('td').get_text(strip=True)

        return {
            'id': book_id,
            'author': author,
            'title': title,
            'summary': summary
        }
    else:
        print(f"Failed to retrieve book {book_id}")
        return None


# Fonction qui va scraper les informations d'une série de livres
def scrape_books(start_id, end_id, save_interval=100):
    books = []

    total_books = end_id - start_id + 1
    for idx, book_id in enumerate(range(start_id, end_id + 1), 1):
        try:
            book_info = get_book_info(book_id)

            if book_info:
                books.append(book_info)
                print(f"Scraped: {book_info['title']} ({idx}/{total_books})")

            if idx % save_interval == 0:
                save_books(books, f'books_data_partial_{book_id}.json')

            time.sleep(1)

        except Exception as e:
            print(f"Erreur pour le livre {book_id}: {e}")
            continue

    save_books(books, f'books_data_final_{end_id}.json')

    return books


# Fonction pour sauvegarder les résultats dans un fichier JSON
def save_books(books, filename="books_data.json"):

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(books, f, ensure_ascii=False, indent=4)
    print(f"Données sauvegardées dans {filename}")


if __name__ == "__main__":
    start_id = 1
    end_id = 7000

    books_data = scrape_books(start_id, end_id)
