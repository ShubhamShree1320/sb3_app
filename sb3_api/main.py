import uvicorn

from sb3_api.app import create_app, get_lifespan
from sb3_api.settings import ServiceSettings

settings = ServiceSettings()
app = create_app(settings=settings, lifespan=get_lifespan)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.SERVICE_PORT)
