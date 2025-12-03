from fastapi import FastAPI
from a2wsgi import ASGIMiddleware
app = FastAPI()
wsgi_app = ASGIMiddleware(app)