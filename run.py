import uvicorn
from gateway.config import LISTEN_HOST, LISTEN_PORT

if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        reload=False,
        proxy_headers=True,
        workers=1,
    )
