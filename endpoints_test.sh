#!/bin/bash
PROD_ENDPOINT_0="https://qywok-tebakaja-proxy-space-0.hf.space"
PROD_ENDPOINT_1="https://qywok-tebakaja-proxy-space-1.hf.space"
PROD_ENDPOINT_2="https://qywok-tebakaja-proxy-space-2.hf.space"
PROD_ENDPOINT_3="https://qywok-tebakaja-proxy-space-3.hf.space"
PROD_ENDPOINT_4="https://qywok-tebakaja-proxy-space-4.hf.space"


check_status() {
  if [ $1 -eq 200 ]; then
    echo "[ Success ] $2 (latency: $3 ms)"
  else
    echo "[ Failed ] $2 (HTTP status: $1)"
  fi
}


make_get_request() {
  START_TIME=$(date +%s%3N)
  RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X GET $1)
  END_TIME=$(date +%s%3N)
  DURATION=$((END_TIME - START_TIME))
  check_status $RESPONSE $2 $DURATION
}


make_post_request() {
  START_TIME=$(date +%s%3N)
  RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST $1 \
    -H "Content-Type: application/json" \
    -d "$2")
  END_TIME=$(date +%s%3N)
  DURATION=$((END_TIME - START_TIME))
  check_status $RESPONSE $3 $DURATION
}


echo "-------- [ Proxy 0 ] --------"
make_get_request "$PROD_ENDPOINT_0/stock/lists" "stock-list-prod-0"
make_post_request "$PROD_ENDPOINT_0/stock/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "stock-prediction-prod-0"
make_get_request "$PROD_ENDPOINT_0/crypto/lists" "crypto-list-prod-0"
make_post_request "$PROD_ENDPOINT_0/crypto/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "crypto-prediction-prod-0"
make_get_request "$PROD_ENDPOINT_0/national-currency/lists" "natcurr-list-prod-0"
make_post_request "$PROD_ENDPOINT_0/national-currency/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "natcurr-prediction-prod-0"
echo " "


echo "-------- [ Proxy 1 ] --------"
make_get_request "$PROD_ENDPOINT_1/stock/lists" "stock-list-prod-1"
make_post_request "$PROD_ENDPOINT_1/stock/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "stock-prediction-prod-1"
make_get_request "$PROD_ENDPOINT_1/crypto/lists" "crypto-list-prod-1"
make_post_request "$PROD_ENDPOINT_1/crypto/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "crypto-prediction-prod-1"
make_get_request "$PROD_ENDPOINT_1/national-currency/lists" "natcurr-list-prod-1"
make_post_request "$PROD_ENDPOINT_1/national-currency/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "natcurr-prediction-prod-1"
echo " "


echo "-------- [ Proxy 2 ] --------"
make_get_request "$PROD_ENDPOINT_2/stock/lists" "stock-list-prod-2"
make_post_request "$PROD_ENDPOINT_2/stock/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "stock-prediction-prod-2"
make_get_request "$PROD_ENDPOINT_2/crypto/lists" "crypto-list-prod-2"
make_post_request "$PROD_ENDPOINT_2/crypto/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "crypto-prediction-prod-2"
make_get_request "$PROD_ENDPOINT_2/national-currency/lists" "natcurr-list-prod-2"
make_post_request "$PROD_ENDPOINT_2/national-currency/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "natcurr-prediction-prod-2"
echo " "


echo "-------- [ Proxy 3 ] --------"
make_get_request "$PROD_ENDPOINT_3/stock/lists" "stock-list-prod-3"
make_post_request "$PROD_ENDPOINT_3/stock/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "stock-prediction-prod-3"
make_get_request "$PROD_ENDPOINT_3/crypto/lists" "crypto-list-prod-3"
make_post_request "$PROD_ENDPOINT_3/crypto/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "crypto-prediction-prod-3"
make_get_request "$PROD_ENDPOINT_3/national-currency/lists" "natcurr-list-prod-3"
make_post_request "$PROD_ENDPOINT_3/national-currency/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "natcurr-prediction-prod-3"
echo " "


echo "-------- [ Proxy 4 ] --------"
make_get_request "$PROD_ENDPOINT_4/stock/lists" "stock-list-prod-4"
make_post_request "$PROD_ENDPOINT_4/stock/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "stock-prediction-prod-4"
make_get_request "$PROD_ENDPOINT_4/crypto/lists" "crypto-list-prod-4"
make_post_request "$PROD_ENDPOINT_4/crypto/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "crypto-prediction-prod-4"
make_get_request "$PROD_ENDPOINT_4/national-currency/lists" "natcurr-list-prod-4"
make_post_request "$PROD_ENDPOINT_4/national-currency/prediction" "{\"days\": 7, \"currency\": \"BTC-USD\"}" "natcurr-prediction-prod-4"
echo " "
