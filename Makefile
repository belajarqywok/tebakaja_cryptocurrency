cutils:
	cd restful/cutils && python setup.py build_ext --inplace && cd ../..

run:
	uvicorn app:app --host 0.0.0.0 --port 7860 --reload