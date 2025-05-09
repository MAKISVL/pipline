import requests
import pandas as pd
from io import StringIO

# Создаем пример DataFrame
sample_data = {

}
df = pd.DataFrame(sample_data)

# Сохраняем DataFrame в CSV
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)
csv_content = csv_buffer.getvalue()

# Отправляем запрос к API
api_url = "usr_2wsFIrXY7IWBr8IhfPmsdl1hOS2/predict/"
files = {'file': ('sample.csv', csv_content, 'text/csv')}
response = requests.post(api_url, files=files)

print(response.json())
