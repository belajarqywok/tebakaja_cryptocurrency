from fastapi import FastAPI
from restful.routes import route
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from restful.cachings import close_caching_connector

app = FastAPI(
    title = "Cryptocurency Prediction Service",
    version = "1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
    allow_credentials = True,
)

app.include_router(
    router = route,
    prefix = '/crypto',
    tags = ['Cryptocurrency']
)

@app.get("/", tags = ['Main'])
def root() -> None:
    return RedirectResponse(url="/docs")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await close_caching_connector()
