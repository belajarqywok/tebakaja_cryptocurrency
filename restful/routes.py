from typing import _ProtocolMeta
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from restful.cachings import connector
from restful.schemas import ForecastingServiceSchema
from restful.controllers import ForecastingControllers


""" API Router """
route = APIRouter()


""" Forecasting Controller """
__CONTROLLER: ForecastingControllers = ForecastingControllers()


""" Caching Connector """
__CONNECTOR: _ProtocolMeta = connector


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
    return await __CONTROLLER.forecasting_controller(payload = payload, caching = __CONNECTOR)

