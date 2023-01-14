import requests

while True:
    me = input('Insa : ')
    url = f'https://api.simsimi.net/v2/?text={me}&lc=id'
    response = requests.get(url)
    if response.status_code == 200:
        print('simi : ', response.json().get('success'))
    else:
        print('bad request!')
