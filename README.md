## Driving Events
Workflow example on the example of a aggressive driving events classifier.


## Data 
The dataset comes from [this reposityory](https://github.com/jair-jr/driverBehaviorDataset)

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

## Run locally

Please download and install Julia 1.5.3. `julia` or `julia.exe` binary should be added to `PATH` variable.

Then in the root of the project type 

`julia --project`
`]instantiate`

Presss backspace and type:

`using Pluto; Pluto.run()`

A new window in browser should appear. Browse for `noteboook/drivingevents.jl`. And you should see the notebook. Wait for a while until all notebooks will be loaded. It may take a little as Julia is compiled language.

## Static HTML

If you don't want to use interactive version just run `notebook/drivingevents.html`

## Online notebook server

The example of running notebook server is at [http://demo-daniel.rnd.grapeup.com/ui/](http://demo-daniel.rnd.grapeup.com/ui/)
