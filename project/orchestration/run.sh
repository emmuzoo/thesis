

#PARAMS=$(sed -e 's/^ *//' < parameters.json | tr -d '\n')

prefect deployment run 'flight-delays-flow/flight-delays-deployment' \
       --params "$(python -c 'import json; import sys; print(json.dumps(json.load(open("params.json"))))')"