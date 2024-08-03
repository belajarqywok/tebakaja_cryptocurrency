from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from restful.controllers import ForecastingControllers
from restful.schemas import ForecastingServiceSchema

""" API Router """
route = APIRouter()

""" Forecasting Controller """
__CONTROLLER = ForecastingControllers()


""" Algorithms Route """
@route.get(path = '/algorithms')
async def algorithms_route() -> JSONResponse:
    return await __CONTROLLER.algorithms_controller()


""" Currencies Route """
@route.get(path = '/currencies')
async def currencies_route() -> JSONResponse:
    return await __CONTROLLER.currencies_controller()


""" Forecasting Route """
@route.post(path = '/forecasting')
async def forecasting_route(
    payload: ForecastingServiceSchema = Body(...)
) -> JSONResponse:
    return await __CONTROLLER.forecasting_controller(payload = payload)

