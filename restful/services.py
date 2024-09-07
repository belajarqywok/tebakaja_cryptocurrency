import json
from typing import _ProtocolMeta
from restful.cutils.utilities import Utilities
from restful.schemas import ForecastingServiceSchema


""" Forecasting Service """
class ForecastingService:

	__FORECAST_UTILS = Utilities()

	async def forecasting(self, payload: ForecastingServiceSchema, caching: _ProtocolMeta) -> dict:
		caching_data = caching.get(
			f'CRYPTO_{payload.algorithm}_{payload.currency}_{payload.days}')

		actuals, predictions = await self.__FORECAST_UTILS.forecasting_utils(
			days        = payload.days,
			algorithm   = payload.algorithm,
			model_name  = payload.currency,

			with_pred   = (caching_data == None),
			sequence_length = 60
		)

		if caching_data != None:
			predictions = json.loads(caching_data.decode('utf-8'))

		else:
			caching.set(
				f'CRYPTO_{payload.algorithm}_{payload.currency}_{payload.days}',
				json.dumps(predictions)
			)

			
		return {'actuals': actuals, 'predictions': predictions}
