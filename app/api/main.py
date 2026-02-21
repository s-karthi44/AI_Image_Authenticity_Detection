from fastapi import FastAPI
from .routes import analyze, status, report

app = FastAPI()

app.include_router(analyze.router)
app.include_router(status.router)
app.include_router(report.router)
