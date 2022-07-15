from datetime import datetime
import imp
from scrapper import haraj_scrapper
from flask import Flask

app = Flask(__name__)

DATES_TYPES = [
    {"query": 'تمر سكري', "type": 'sukkari'},
    {"query": 'تمر مجدول', "type": 'medjool'},
    {"query": 'تمر شيشي', "type": 'shaishe'},
    {"query": 'تمر نبة علي', "type": 'nabtat ali'},
    {"query": 'تمر صقعي', "type": 'sugaey'},
    {"query": 'تمر عجوة', "type": 'ajwa'},
    {"query": 'تمر خلاص', "type": 'khulas'},
]
@app.route("/")
def hello_world():
    for date in DATES_TYPES:
        # Run HARAJ Scrapper
        haraj_scrapper(date['query'], date['type'])
    print(f'Scrapping Finished on {datetime.now()}')
