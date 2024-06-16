## Запуск приложения
> [!CAUTION]
> Ветки **main** и **staging** используются для развертывания приложения на нашей ВМ.
> Чтобы запустить проект локально, необходимо использовать ветку **demo**

Необходимые ENV переменные указаны в .env файле

Для запуска проекта необходим установленный на системе docker
```commandline
docker compose up --build
```

- WEB-cервис будет доступен по адресу <localhost:8521>
- Swagger API будет доступен по адресу <localhost:8520/docs>

## Запущенное нами приложение
Docker-compose развернут на удаленной ВМ Yandex Cloud

- WEB-cервис доступен по адресу <http://punkbutterfly.tech:8511>
- Swagger API доступен по адресу <http://punkbutterfly.tech:8510/docs>

## Структура проекта
```
├── .github -> CI/CD пайплайны  
├── backend -> Директория backend микросервиса
│  ├── models
│  │  ├── weights -> Веса моделей
│  │  ├── Detector.py -> Инференс модели детекции
│  ├── api.py -> backend приложение 
│  ├── Dockerfile
│  ├── requirements.txt 
├── frontend -> Директория frontend микросервиса
│  ├── app.py -> frontend приложение
│  ├── Dockerfile
│  ├── requirements.txt 
├── notebooks -> Ноутбуки подготовки данных и обучения моделей
├── .env -> Переменные окружения, необходимые для запуска проекта
├── .gitignore
├── docker-compose.yaml
├── README.md  
```