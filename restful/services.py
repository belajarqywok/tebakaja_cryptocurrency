from restful.cutils.utilities import Utilities
from restful.schemas import ForecastingServiceSchema


""" Forecasting Service """
class ForecastingService:

	__FORECAST_UTILS = Utilities()

	async def forecasting(self, payload: ForecastingServiceSchema) -> dict:
		days: int      = payload.days
		currency: str  = payload.currency
		algorithm: str = payload.algorithm

		actuals, predictions = await self.__FORECAST_UTILS.forecasting_utils(
			days            = days,
			algorithm       = algorithm,
			model_name      = currency,

			sequence_length = 60
		)

		return {'actuals': actuals, 'predictions': predictions}
