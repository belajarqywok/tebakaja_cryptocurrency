from pydantic import BaseModel

class ForecastingServiceSchema(BaseModel) :
	days: int
	currency: str
	algorithm: str
	