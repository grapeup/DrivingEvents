## Driving Events
Aggressive driving events classifier - workflow example.


## Data 
The dataset comes from [this repository](https://github.com/jair-jr/driverBehaviorDataset).

Please download the data and create following file structure:
```
data
├── cartrip1
│   ├── aceleracaoLinear_terra.csv
│   ├── acelerometro_terra.csv
│   ├── campoMagnetico_terra.csv
│   ├── giroscopio_terra.csv
│   ├── groundTruth.csv
│   └── viagem.json
├── cartrip2
│   ├── aceleracaoLinear_terra.csv
│   ├── acelerometro_terra.csv
│   ├── campoMagnetico_terra.csv
│   ├── giroscopio_terra.csv
│   ├── groundTruth.csv
│   └── viagem.json
├── cartrip3
│   ├── aceleracaoLinear_terra.csv
│   ├── acelerometro_terra.csv
│   ├── campoMagnetico_terra.csv
│   ├── giroscopio_terra.csv
│   ├── groundTruth.csv
│   └── viagem.json
└── cartrip4
    ├── aceleracaoLinear_terra.csv
    ├── acelerometro_terra.csv
    ├── campoMagnetico_terra.csv
    ├── giroscopio_terra.csv
    ├── groundTruth.csv
    └── viagem.json
```

The `data` directory should be placed in root of the project (this file location).

## Run interactive notebook locally

Please [download](https://julialang.org/downloads/) and install Julia 1.5.3. `julia` or `julia.exe` binary should be added to `PATH` environment variable.

Then in the root of the project type:

`julia --project`

And then:

`]instantiate`

Presss backspace and type:

`using Pluto; Pluto.run()`

A new window in your browser should appear. Browse for `noteboook/drivingevents.jl`. You should see the notebook. Wait for a while until all sections will be loaded. It may take some time as Julia is compiled language.

## Static HTML

If you don't want to use interactive version just run `notebook/drivingevents.html`

## Online notebook

Static HTML version of the notebook can be also found online under this link: [http://demo-daniel.rnd.grapeup.com/notebook](http://demo-daniel.rnd.grapeup.com/notebook).

## Mlflow

Mlflow logs can be found [here](http://18.185.244.61:5050/#/experiments/3).

## Run server locally

Please [download](https://julialang.org/downloads/) and install Julia 1.5.3. `julia` or `julia.exe` binary should be added to `PATH` environment variable.

Then in the root of the project type:

`sh bin/setup`
`sh bin/server`

## Run docker

Please install docker with docker-compose tool.

Then in the root of the project type:

`sh dockerbuild.sh`
`sh dockerrun.sh`

## Non-technical presentation

The presentation can be found in the root folder, filename: DrivingEvents.pdf.
