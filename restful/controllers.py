import os
from http import HTTPStatus
from typing import _ProtocolMeta
from fastapi.responses import JSONResponse
from restful.services import ForecastingService
from restful.schemas import ForecastingServiceSchema


# Forecasting Controller
class ForecastingControllers:

    __SERVICE: ForecastingService = ForecastingService()


    # Algorithms Controller
    async def algorithms_controller(self) -> JSONResponse:
        try:
            algorithms: list = sorted(
                os.listdir("resources/algorithms"))

            return JSONResponse(
                content = {
                    'message': 'Success',
                    'status_code': HTTPStatus.OK,
                    'data': algorithms
                },
                status_code = HTTPStatus.OK
            )

        except Exception as error_message:
            print(error_message)
            return JSONResponse(
                content = {
                    'message': 'Internal Server Error',
                    'status_code': HTTPStatus.INTERNAL_SERVER_ERROR,
                    'data': None
                },
                status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            )



    # Currency Controller
    async def currencies_controller(self) -> JSONResponse:
        try:
            path: str = 'resources/datasets'
            datasets: list = sorted(
                [
                    item.replace(".csv", "") for item in os.listdir(path)
                    if os.path.isfile(os.path.join(path, item)) 
                        and item.endswith('.csv')
                ]
            )

            return JSONResponse(
                content = {
                    'message': 'Success',
                    'status_code': HTTPStatus.OK,
                    'data': datasets
                },
                status_code = HTTPStatus.OK
            )

        except Exception as error_message:
            print(error_message)
            return JSONResponse(
                content = {
                    'message': 'Internal Server Error',
                    'status_code': HTTPStatus.INTERNAL_SERVER_ERROR,
                    'data': None
                },
                status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            )



    # Forecasting Controller
    async def forecasting_controller(self, payload: ForecastingServiceSchema,
        caching: _ProtocolMeta) -> JSONResponse:
        try:
            path: str = 'resources/datasets'
            datasets: list = sorted(
                [
                    item.replace(".csv", "") for item in os.listdir(path)
                    if os.path.isfile(os.path.join(path, item)) and item.endswith('.csv')
                ]
            )

            if payload.currency not in datasets:
                return JSONResponse(
                    content = {
                        'message': f'symbols "{payload.currency}" is not available.',
                        'status_code': HTTPStatus.BAD_REQUEST,
                        'data': None
                    },
                    status_code = HTTPStatus.BAD_REQUEST
                )


            if (payload.days > 31) or (payload.days < 1):
                return JSONResponse(
                    content = {
                        'message': 'days cannot be more than a month and cannot be less than 1',
                        'status_code': HTTPStatus.BAD_REQUEST,
                        'data': None
                    },
                    status_code = HTTPStatus.BAD_REQUEST
                )


            if payload.algorithm not in os.listdir("resources/algorithms"):
                return JSONResponse(
                    content = {
                        'message': f'algorithm "{payload.algorithm}" is not available.',
                        'status_code': HTTPStatus.BAD_REQUEST,
                        'data': None
                    },
                    status_code = HTTPStatus.BAD_REQUEST
                )


            prediction: dict = await self.__SERVICE.forecasting(
                payload = payload, caching = caching)

            if not prediction :
                return JSONResponse(
                    content = {
                        'message': 'prediction could not be generated, please try again.',
                        'status_code': HTTPStatus.BAD_REQUEST,
                        'data': None
                    },
                    status_code = HTTPStatus.BAD_REQUEST
                )

            return JSONResponse(
                content = {
                    'message': 'prediction success',
                    'status_code': HTTPStatus.OK,
                    'data': {
						'currency': payload.currency,
						'predictions': prediction
					}
                },
                status_code = HTTPStatus.OK
            )

        except Exception as error_message:
            print(error_message)
            return JSONResponse(
                content = {
                    'message': 'internal server error',
                    'status_code': HTTPStatus.INTERNAL_SERVER_ERROR,
                    'data': None
                },
                status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            )
