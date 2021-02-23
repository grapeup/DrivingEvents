### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ ed55f87e-6ebc-11eb-0e31-539f6c4616cc
begin
	using Pkg
	rootdir = joinpath(@__DIR__, "..")
	Pkg.activate(rootdir)
	Pkg.instantiate()
	cd(rootdir)
	md"project root and current directory: $(pwd())"
end

# ╔═╡ d3df7b06-6eba-11eb-0985-e9e458ee8dc8
begin
	using StatsPlots
	using Dates
	using JSON
	using CSV
	using DataFrames
	using Gadfly
	using Genie
	using PlutoUI 
	using PrintFileTree
	using OrderedCollections
	using Revise
	using Statistics
	using PyCall
	using RollingFunctions
	using LinearAlgebra
	using MLLabelUtils
	using ScikitLearn
	using ScikitLearn.CrossValidation: 
		cross_val_score, train_test_split, StratifiedKFold
	using DecisionTree
	using Distributions
	using DrivingEvents
end

# ╔═╡ 644a7e0a-6eb8-11eb-261c-9feb1fe2b353
md"""
### Driving Events
##### Introduction
This notebook is created by Daniel Bulanda, machine learning engineer at Grape Up, and was made as a part of the technical evaluation process for Hertz. The goal is to create aggressive driving events classifier.
"""

# ╔═╡ 95d96c92-6ebd-11eb-136d-1d3a3f7f0893
md"##### Project startup"

# ╔═╡ b2283ab8-6ebd-11eb-31e3-5770050b62b6
md"Activate the project, download required packages, and go set proper pwd."

# ╔═╡ 7b591c3a-6ebf-11eb-1824-27b2057eda6e
md"Julia imports."

# ╔═╡ d989fcaa-7434-11eb-3223-0935342ccefd
md"Python imports."

# ╔═╡ 9185384e-7393-11eb-2be3-dba24541dfbf
begin
	mlflow = pyimport("mlflow")
	
	@sk_import utils.class_weight: compute_class_weight
	@sk_import metrics: confusion_matrix
	@sk_import metrics: accuracy_score
	@sk_import metrics: make_scorer
	@sk_import metrics: classification_report
	@sk_import naive_bayes: GaussianNB
	@sk_import neighbors: KNeighborsClassifier
	@sk_import ensemble: RandomForestClassifier
	nothing
end

# ╔═╡ 5d96ac50-7593-11eb-3686-b9be36643e28
md"Setup mlflow."

# ╔═╡ 54a25860-7593-11eb-0dc8-c189891997f7
begin
	remote_server_uri = "http://18.185.244.61:5050"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("DrivingEvents") 
    mlflow.end_run()
    mlflow.start_run(run_name = "")
end

# ╔═╡ c71223e6-715c-11eb-3aa0-49eef018dde0
md"Configure plotting in notebook."

# ╔═╡ f79fe3b0-715c-11eb-0517-4ff69996aca5
begin
	gr(display_type=:inline)
	gr(size=(700, 550))
	gr(fmt=:png)
	nothing
end

# ╔═╡ cb9cc790-6ec0-11eb-1163-8301e347a97f
md"##### Data preparation"

# ╔═╡ 7686accc-7027-11eb-0917-1db37660474c
md"""
With the permission of the author, the dataset comes from [this repository](https://github.com/jair-jr/driverBehaviorDataset).

The dataset is a collection of smartphone sensor measurements for driving events.
"""

# ╔═╡ 61fe483e-7034-11eb-13fa-3b6e28ce2c30
md"The `data` directory contains 4 car trips of approximately 13 minutes each, in average."

# ╔═╡ 157566bc-7022-11eb-3d43-697e6d009061
with_terminal() do
	printfiletree("data")
end

# ╔═╡ 11132b80-703a-11eb-1b79-d376afe6099e
md"Collect the raw data in appropriate data structures and a quick look at representative examples."

# ╔═╡ e1bbb95a-6ec0-11eb-2c5a-917b55d9ebae
begin
	rawdata = OrderedDict{Symbol, OrderedDict{Symbol, Any}}()
	
	for i in 1:4
		rawdata[Symbol("trip$i")] = Dict{String, DataFrame}()
		
		rawdata[Symbol("trip$i")][:acclin] = 
			CSV.File("data/cartrip$i/aceleracaoLinear_terra.csv") |> DataFrame
		
		rawdata[Symbol("trip$i")][:acc] = 
			CSV.File("data/cartrip$i/acelerometro_terra.csv") |> DataFrame
		
		rawdata[Symbol("trip$i")][:mag] = 
			CSV.File("data/cartrip$i/campoMagnetico_terra.csv") |> DataFrame
		
		rawdata[Symbol("trip$i")][:gyro] = 
			CSV.File("data/cartrip$i/giroscopio_terra.csv") |> DataFrame
		
		rawdata[Symbol("trip$i")][:gt] = 
			CSV.File("data/cartrip$i/groundTruth.csv") |> DataFrame
		
		rawdata[Symbol("trip$i")][:meta] = 
			JSON.parse(read(open("data/cartrip$i/viagem.json", "r"), String))
	end
	
	rawdata
end

# ╔═╡ b1ecfbde-7209-11eb-2d7b-79a50fe9ae91
md"""
Abbreviations explanation and units:
- `acclin` - linear acceleration [m/s2],
- `acc` - accelerometer [m/s2],
- `mag` - magnetometer [μT],
- `gyro` - gyroscope [rad/s].
"""

# ╔═╡ c149e238-704a-11eb-3b34-e165bb13e4cc
md"The summary of each data table."

# ╔═╡ 9d22af32-7060-11eb-073f-4debb0fd8597
begin
	rawdatainfo = OrderedDict{Symbol, OrderedDict{Symbol, Any}}()
	
	for i in 1:4
		rawdatainfo[Symbol("trip$i")] = Dict{String, DataFrame}()
		
		for tablename in [:acclin, :acc, :mag, :gyro, :gt]
			rawdatainfo[Symbol("trip$i")][tablename] = 
				describe(rawdata[Symbol("trip$i")][tablename])
		end
	end
	
	rawdatainfo
end

# ╔═╡ 6f5c89ca-705f-11eb-3028-7d8f072b8e32
md"There are no missing values in the dataset."

# ╔═╡ 8586aeb8-705f-11eb-1620-2df3b3002608
md"Sampling rate analysis."

# ╔═╡ a00fd488-7081-11eb-1e23-e1d1011d9c8a
md"The magnetometer samples with close to double the frequency of the other sensors. The problem is that `number of samples for magnetometer / 2` is close but not equal to `number of samples for other sensors`, so halving the number of the magnetometer signals doesn't solve the problem. Here is a function which picks `n` equally distributed indices from a `range`."

# ╔═╡ 81ba6b40-70ec-11eb-2369-c76fb4390f60
function eqdist(source, target)
	idxsrange = 1:target
	factor = source / target
	round.(Int, collect(idxsrange) .* factor)
end

# ╔═╡ e9eaacda-70e6-11eb-04f4-69c351839996
eqdist(nrow(rawdata[:trip1][:mag]), nrow(rawdata[:trip1][:acc]))

# ╔═╡ 7ac967d2-70e7-11eb-1af1-a9d8e82c80fc
md"The `eqdist` function solved the problem."

# ╔═╡ eb944420-7063-11eb-3188-533bd3f2d3ed
md"Transformations of the original data will be stored in `data` dictionary."

# ╔═╡ 50b5980e-70ff-11eb-3db0-67d1bc0d4361
begin
	data = OrderedDict{Symbol, Any}()
	
	for tripname in keys(rawdata)
		data[tripname] = Dict{Symbol, DataFrame}()
		for tablename in [:acclin, :acc, :mag, :gyro]
			data[tripname][tablename] = rawdata[tripname][tablename][:, :]
		end
	end
end

# ╔═╡ 7b5c15ec-704b-11eb-1dbe-373102452d5b
begin
	rawcounters = DataFrame(name = String[], count = Int[])
	
	for (tripname, trip) in data
		for (tablename, table) in trip
			if tablename in [:acc, :acclin, :mag, :gyro]
				push!(rawcounters, ("$tripname $tablename", nrow(table)))
			end
		end
	end
	
	rawcounters
end

# ╔═╡ cc76eb6e-70ff-11eb-26f7-89e592f0187c
md"Apply the `eqdist` function."

# ╔═╡ 18e7214e-7083-11eb-2dab-2b40ff9eee3d
begin
	datacounters = DataFrame(name = String[], count = Int[])

	for trip in keys(data)
		indices = eqdist(nrow(rawdata[trip][:mag]), nrow(rawdata[trip][:acc]))
		data[trip][:mag] = rawdata[trip][:mag][indices, :]
		for (tablename, table) in data[trip]
			push!(datacounters, ("$trip $tablename", nrow(table)))
		end
	end
	
	datacounters
end

# ╔═╡ fea57d42-70eb-11eb-1249-8bd9af09bfcb
md"Now all recordings have the same sampling rate among the car trips."

# ╔═╡ 2cd75a90-70fe-11eb-314f-a9b054936350
md"Only the `timestamp` columns require conversion from `String` to `DateTime`. "

# ╔═╡ 4896f82e-7065-11eb-220e-917923327763
begin
	dtformat = DateFormat("dd/mm/yyyy HH:MM:SS")
	typesok = true
	
	for tripname in keys(rawdata)
		for tablename in [:acclin, :acc, :mag, :gyro]
			timestampcol = data[tripname][tablename][!, :timestamp]
			
			if eltype(timestampcol) == String
				data[tripname][tablename][!, :timestamp] = 
					DateTime.(timestampcol, dtformat)
			end
			
			if eltype(data[tripname][tablename][!, :timestamp]) != DateTime
				global typesok = false
			end
		end
	end

	typesok ? md"timestamps converted successfully" : md"timestamps conversion failed"
end

# ╔═╡ b5f31a70-70f7-11eb-3060-e5ec5cc2ae6b
md"I want to check if recordings are synchronized among car trips by comparing start time."

# ╔═╡ 42f072b6-70f0-11eb-1d11-19c6d66e286c
begin
	syncok = true
	
	for (tripname, trip) in data
		startdate = trip[:acc][1, :timestamp]
		for (tablename, table) in trip
			if startdate != table[1, :timestamp]
				global syncok = false
			end
		end
	end
	
	md"start time synchronized: $syncok"
end

# ╔═╡ 1a9bae16-70f0-11eb-1bd0-dfeb3e8adb08
md"Now I want to add a label for each row of the recordings. To do so I need to add a a new column for recordings containing seconds elapsed since the start of the trip, because that's the way the authors of the dataset stored information about target classes."

# ╔═╡ 280c9254-70fa-11eb-30d9-c57870b21c0a
begin
	startstopsync = DataFrame(
		name = String[], 
		absstart = Int[],
		absstop = Int[],
		relstart = Float64[],
		relstop = Float64[]
	)
	
	for (tripname, trip) in data
		for (tablename, table) in trip
			startdate = table[1, :timestamp]
			startnano = table[1, :uptimeNanos]
			
			sssabs = convert.(
				Int, 
				(table[!, :timestamp] .- startdate) / Millisecond(1) * (1 / 1000)
			)
			sssrel = 
				(table[!, :uptimeNanos] .- startnano) ./ 10^9
			
			push!(startstopsync, ("$tripname $tablename",
				sssabs[1],
				sssabs[end],
				sssrel[1],
				sssrel[end]
			))
			
			table.sss = sssrel
		end
	end
	
	startstopsync
end

# ╔═╡ b6b5ae14-7109-11eb-225e-b34060198e55
md"There are small differences between absolute and relative timer, but it's acceptable because sensors sampling is not synchronized and there is a small gap caused by `timestamp` column accuracy in seconds, while `uptimeNanos` column is in nanoseconds. For labeling `sss` column is added to the data which stands for `seconds from start`."

# ╔═╡ 4bf118b0-7119-11eb-10fc-8390fa2b49a3
md"Now I can easily check sampling frequency for each trip."

# ╔═╡ 59a62a10-7144-11eb-2da1-ed4d5b9dce07
begin
	samplingrates = DataFrame(name = String[], hz = Float64[])

	for (tripname, trip) in data
		for (tablename, table) in trip
			freq = nrow(table) / table[end, :sss]
			push!(samplingrates, ("$tripname $tablename", freq))
		end
	end
	
	samplingrates
end

# ╔═╡ 12ac1eac-7148-11eb-0c8d-1de16d04534d
md"Probably the desired sampling frequency is 50 Hz. Anyway, this small difference shouldn't matter to our problem."

# ╔═╡ e60ec0ea-711a-11eb-29cf-b91ecdb47d87
md"Aggregate all trips data into one table. A postfix `x`, `y` or `z` which represents the respective axes in the Cartesian coordinate system, is appended at the end of given column name where a prefix is the sensor short name explained above."

# ╔═╡ 95372af0-710a-11eb-0468-759d65813306
for (tripname, trip) in data
	tripdf = trip[:all] = DataFrame()

	tripdf.sss = trip[:acc][!, :sss]

	tripdf.acclinx = trip[:acclin][!, :x]
	tripdf.accliny = trip[:acclin][!, :y]
	tripdf.acclinz = trip[:acclin][!, :z]

	tripdf.accx = trip[:acc][!, :x]
	tripdf.accy = trip[:acc][!, :y]
	tripdf.accz = trip[:acc][!, :z]

	tripdf.gyrox = trip[:gyro][!, :x]
	tripdf.gyroy = trip[:gyro][!, :y]
	tripdf.gyroz = trip[:gyro][!, :z]

	tripdf.magx = trip[:mag][!, :x]
	tripdf.magy = trip[:mag][!, :y]
	tripdf.magz = trip[:mag][!, :z]
end

# ╔═╡ 03fb8700-7125-11eb-04eb-61e2e3213b30
md"Mapping labels from Portuguese to English."

# ╔═╡ e81d7a0c-7124-11eb-32fa-c7d4611e9fa4
eventmapping = Dict(
	"aceleracao_agressiva" => "aggressive_acceleration",
	"freada_agressiva" => "aggressive_braking",
	"curva_direita_agressiva" => "aggressive_right_turn",
	"curva_esquerda_agressiva" => "aggressive_left_turn",
	"troca_faixa_direita_agressiva" => "aggressive_right_lane_change",
	"troca_faixa_esquerda_agressiva" => "aggressive_left_lane_change",
	"evento_nao_agressivo" => "non-aggressive_event",
)

# ╔═╡ 20fa98e2-7188-11eb-0ec9-aba67380afa8
md"List of possible labels."

# ╔═╡ 8f6e7c22-7187-11eb-23d3-157713f21e0b
labels = sort(["none", values(eventmapping)...])

# ╔═╡ f8a9db0e-711a-11eb-0013-0fb1ff1a02e1
md"Add label to each per-trip aggregated data."

# ╔═╡ 19bc4e3a-711b-11eb-0fb9-915e415d545d
for (tripname, trip) in data
	evcol = trip[:all].event = Vector{String}(undef, nrow(trip[:all]))
	evcol .= "none"
	
	for event in eachrow(rawdata[tripname][:gt])
		evstart = event[" inicio"]
		evend = event[" fim"]
		evname = eventmapping[event["evento"]]
		sss = trip[:all][!, :sss]
		
		evidxs = (sss .>= evstart) .& (sss .<= evend)
		evcol[evidxs] .= evname
	end
end

# ╔═╡ dda67424-7148-11eb-00a8-75807d6c825d
md"And finally, merge all the trips into one table, with \"flatten\" `sss` column to concatenate all trips into one big trip. That step is OK because attributes will be treated as time-independent in the next steps."

# ╔═╡ 42780f5c-7149-11eb-3461-bb606acb1d67
let
	trip1 = data[:trip1][:all][:, :]
	
	trip2 = data[:trip2][:all][:, Not(:sss)]
	trip2.sss = data[:trip2][:all][:, :sss] .+ trip1[end, :sss]
	
	trip3 = data[:trip3][:all][:, Not(:sss)]
	trip3.sss = data[:trip3][:all][:, :sss] .+ trip2[end, :sss]

	trip4 = data[:trip4][:all][:, Not(:sss)]
	trip4.sss = data[:trip4][:all][:, :sss] .+ trip3[end, :sss]
	
	global data[:all] = [trip1; trip2; trip3; trip4]
end

# ╔═╡ 012af6d4-714b-11eb-1b7f-e92ef44b646d
md"##### Exploratory Data Analysis"

# ╔═╡ 3cfc5fbc-7151-11eb-3ea0-f558fb038e29
md"Lets start with overview of available examples for each class."

# ╔═╡ 61bfcc94-7151-11eb-2534-d94a806213c9
begin
	classes = combine(groupby(data[:all], :event), nrow)
	sort!(classes)
	classes.events = zeros(Int, nrow(classes))
	classes.meantime_s = zeros(Float64, nrow(classes))
	classes.mintime_s = Vector{Float64}(undef, nrow(classes)) .= typemax(Int)
	classes.maxtime_s = zeros(Float64, nrow(classes))
	
	counter = 0
	lastevent = "none"
	hz = samplingrates[1, :hz]
	
	for row in eachrow(data[:all])
		counter += 1
		currentev = row[:event]
		
		if lastevent != currentev
			evidxs = classes[!, :event] .== lastevent
			
			classes[evidxs, :events] .+= 1
			
			elapsed = round(counter / hz, digits = 2)
			classes[evidxs, :maxtime_s] .= 
				max(first(classes[evidxs, :maxtime_s]), elapsed)
			classes[evidxs, :mintime_s] .= 
				min(first(classes[evidxs, :mintime_s]), elapsed)
			
			lastevent = currentev
			counter = 1
		end
	end
	
	classes[classes.event .== "none", :events] .+= 1
	eventsno = sum(classes.events)
	classes[!, :meantime_s] = round.(classes.nrow ./ classes.events ./ hz, digits = 2)
	
	classes
end

# ╔═╡ 3cbb0d64-7156-11eb-3c81-1f9c6044ccb8
begin
	evetnsno = sum(classes[classes.event .!= "none", :events])
	gapsno = evetnsno + 1
	gapfactor = round(
		Int, 
		100 * first(classes[classes.event .== "none", :nrow]) / nrow(data[:all])
	)
	gapmin = first(classes[classes.event .== "none", :mintime_s])
	gapmax = first(classes[classes.event .== "none", :maxtime_s])
	gapmeam = first(classes[classes.event .== "none", :meantime_s])
	md"There are $evetnsno recorded events and $gapsno \"gaps\" between them. The number of lane change events is relatively smaller. It's worth noting that $gapfactor % of driving time no event was recorded and gaps durations are in the range from $gapmin s to $gapmax s (mean $gapmeam s)."
end

# ╔═╡ 9ecf1b06-727f-11eb-3d99-310b3ff751d3
begin
	meanevtime = round(mean(
		classes[classes.event .!= "none", :meantime_s]), 
		digits = 2
	)
	minevtime = min(classes[classes.event .!= "none", :mintime_s]...)
	maxevtime = max(classes[classes.event .!= "none", :maxtime_s]...)
	
	md"""
	Summary of events duration:
	- Mean event duration is $(meanevtime) s.
	- The shortest event duration is $(minevtime) s.
	- The longest event duration is $(maxevtime) s.
	These values are important for setting best sliding window width in the future steps.
	"""
end

# ╔═╡ de974a20-7150-11eb-05b4-497bcf6c345c
md"Summary of the numerical data."

# ╔═╡ f17799c6-7150-11eb-17c5-a5ea424cbb39
describe(data[:all][!, Not(:event)])

# ╔═╡ 07009f06-71fe-11eb-24b4-134600e50d86
md"The linear acceleration `acclin` is similar to the accelerometer `acc`, but excluding the force of gravity. Hence statistics for them are the almost same except z-axis where `mean(accz) ≈ 9.7` vs  `mean(acclinz) ≈ -0.1` (the gravity acceleration is about 9,81 m/s2)."

# ╔═╡ 46aa540e-751d-11eb-1b77-23f3adc6c7fd
md"The `magx` values are relatively small, comparing to other values."

# ╔═╡ 2f4095de-7159-11eb-2945-254c26081516
md"Total driving duration is about $(round(Int, data[:all][end, :sss] / 60)) minutes."

# ╔═╡ 26d7e2ca-715a-11eb-0b75-458e78fec242
@df data[:all] cornerplot(
	cols(2:13), 
	grid = false, 
	compact = true, 
	xticks = [], 
	yticks = [], 
	xtickfontsize = 1, 
	ytickfontsize = 1, 
	yguidefontsize = 7, 
	xguidefontsize = 7, 
	margin = 0mm,
	dpi = 200
)

# ╔═╡ c0be143c-7172-11eb-1f87-55db449dc586
md"Nothing really interesting in correlation matrix. There is an obvious correlation between accelerometer and linear acceleration for respective axes, and some weak and not important correlations between various sensors. The data distribution on histograms looks very good, except `magy`."

# ╔═╡ 97c57b7c-7188-11eb-0619-8d900fbe23bc
md"Sensors data visualization for aggresive right turn. The plot shows 2 s before and after 2 s this event."

# ╔═╡ 78f9b418-72ac-11eb-2822-17db36df1c60
begin
	function sensorplot(col::Symbol, title::String, xrange::UnitRange{Int})
		accxplot = Gadfly.plot(
			data[:all][xrange, :], 
			x = :sss, 
			y = col, 
			color = :event, 
			Geom.line,
			style(key_position = :top),
			Guide.xticks(ticks=nothing),
			Guide.xlabel("time"),
			Guide.ylabel(nothing),
			Guide.colorkey(title = title)
		)
	end
	
	set_default_plot_size(18cm, 24cm)
	vstack(
		hstack(
			sensorplot(:acclinx, "linear acceleration x-axis", 896:1298),
			sensorplot(:accliny, "linear acceleration y-axis", 896:1298),
			sensorplot(:acclinz, "linear acceleration z-axis", 896:1298),
		),
		
		hstack(
			sensorplot(:accx, "accelerometer x-axis", 896:1298),
			sensorplot(:accy, "accelerometer y-axis", 896:1298),
			sensorplot(:accz, "accelerometer z-axis", 896:1298),
		),
		
		hstack(
			sensorplot(:gyrox, "gyroscope x-axis", 896:1298),
			sensorplot(:gyroy, "gyroscope y-axis", 896:1298),
			sensorplot(:gyroz, "gyroscope z-axis", 896:1298),
		),
		
		hstack(
			sensorplot(:magx, "magnetometer x-axis", 896:1298),
			sensorplot(:magy, "magnetometer y-axis", 896:1298),
			sensorplot(:magz, "magnetometer z-axis", 896:1298),
		)
	)
end

# ╔═╡ 12a8e618-72be-11eb-1b73-bb0bcc914d37
md"In this example linear acceleration, accelerometer and gyroscope change their indications clearly in all axes and returns to their previous indications. Magnetometer seems to be useless for x-axis, and probably for other axes because it measures orientation in earth's magnetic field which is probably not very specific for events we are going to detect but rather for the car's position relative to the ground."

# ╔═╡ 1144b95c-7190-11eb-161e-ab808884e4a4
begin
    evidxs = data[:all].event .!= "none"
    evdata = data[:all][evidxs, :]
    StatsPlots.plot(
        @df(evdata, density(
            :acclinx, 
            group = (:event), 
            legend = :outertopleft,
            grid = false,
            showaxis = false,
            ticks = [],
            linewidth = 0,
            legendfontsize = 9,
        )),
        @df(evdata, density(
            :acclinx, 
            legend = :none,
            grid = false,
            showaxis = false,
            ticks = [],
            linewidth = 0,
            title = "sensors values densities grouped by events",
            titlefontsize = 12,
        )),
        @df(evdata, density(
            :acclinx, 
            legend = :none,
            grid = false,
            showaxis = false,
            ticks = [],
            linewidth = 0,
        )),
        
        @df(evdata, density(
            :acclinx, 
            group = (:event), 
            legend = :none, 
            title = "acclinx",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        @df(evdata, density(
            :accliny, 
            group = (:event), 
            legend = :none, 
            title = "accliny",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        @df(evdata, density(
            :acclinz, 
            group = (:event), 
            legend = :none, 
            title = "acclinz",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        
        @df(evdata, density(
            :accx, 
            group = (:event), 
            legend = :none, 
            title = "accx",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        @df(evdata, density(
            :accy, 
            group = (:event), 
            legend = :none, 
            title = "accy",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        @df(evdata, density(
            :accz, 
            group = (:event), 
            legend = :none, 
            title = "accz",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        
        @df(evdata, density(
            :gyrox, 
            group = (:event), 
            legend = :none, 
            title = "gyrox",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        @df(evdata, density(
            :gyroy, 
            group = (:event), 
            legend = :none, 
            title = "gyroy",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        @df(evdata, density(
            :gyroz, 
            group = (:event), 
            legend = :none, 
            title = "gyroz",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        
        @df(evdata, density(
            :magx, 
            group = (:event), 
            legend = :none, 
            title = "magx",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        @df(evdata, density(
            :magy, 
            group = (:event), 
            legend = :none, 
            title = "magy",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        @df(evdata, density(
            :magz, 
            group = (:event), 
            legend = :none, 
            title = "magz",
            titlefontsize = 10,
			xtickfontsize = 5, 
			ytickfontsize = 5,
        )),
        
        layout = (5, 3),
        size = (700, 1100),
        dpi = 200
    )
end

# ╔═╡ 3cf876a4-727d-11eb-248c-db2f5a822c21
md"The density distribution between classes looks promising for most of the features. Probably, the target classes can be described well using some of them. Both acceleration and gyroscope seem to have good predictive power. On `magx` plot most of the classes looks very similar, so probably it won't be a usefull feature. I have mixed feelings about `magy` and `magz` but their importance will be assesed by models in the next steps. It is worth noting that non-aggresive events can be potentially a problematic class because of its ambiguity."

# ╔═╡ 02a4bef6-72cd-11eb-2387-2783e5833851
begin
	indices = Dict()
	indices[:aa] = (
		data[:all].event .== "aggressive_acceleration", 
		"aggresive acceleration"
	)
	indices[:br] = (
		data[:all].event .== "aggressive_braking", 
		"aggressive breaking"
	)
	indices[:ll] = (
		data[:all].event .== "aggressive_left_lane_change", 
		"aggressive left lane change"
	)
	indices[:rl] = (
		data[:all].event .== "aggressive_right_lane_change", 
		"aggressive right lane change"
	)
	indices[:lt] = (
		data[:all].event .== "aggressive_left_turn", 
		"aggressive left turn"
	)
	indices[:rt] = (
		data[:all].event .== "aggressive_right_turn", 
		"aggressive right turn"
	)
	indices[:na] = (
		data[:all].event .== "non-aggressive_event", 
		"non-aggressive event"
	)
	indices[:no] = (data[:all].event .== "none", "none")
	indices
end

# ╔═╡ f641dc68-72c0-11eb-1eca-fd4af026a286
begin
	Plots.plot(
		data[:all][indices[:aa][1], :acclinx], 
		data[:all][indices[:aa][1], :accliny], 
		data[:all][indices[:aa][1], :acclinz],
		seriestype=:scatter, 
		markersize = 1,
		markerstrokewidth = 0,
		smooth = true,
		title = "linear acceleration xyz-axes",
		titlefontsize = 12,
		xlims = (-5, 5),
		ylims = (-5, 5),
		zlims = (-5, 5),
		color = :darktest,
		labels = indices[:aa][2],
		dpi = 200
	)
	
	function acclin2plot(idxs, title)
		Plots.plot!(
			data[:all][idxs, :acclinx], 
			data[:all][idxs, :accliny], 
			data[:all][idxs, :acclinz],
			seriestype=:scatter, 
			markersize = 1,
			markerstrokewidth = 0,
			smooth = true,
			labels = title
		)
	end
	
	acclin2plot(indices[:br][1], indices[:br][2])
	acclin2plot(indices[:ll][1], indices[:ll][2])
	acclin2plot(indices[:rl][1], indices[:rl][2])
	acclin2plot(indices[:lt][1], indices[:lt][2])
	acclin2plot(indices[:rt][1], indices[:rt][2])
	acclin2plot(indices[:na][1], indices[:na][2])
end

# ╔═╡ 66c1445c-72d2-11eb-2a05-3fdb04a1d92b
begin
	Plots.plot(
		data[:all][indices[:aa][1], :accx], 
		data[:all][indices[:aa][1], :accy], 
		data[:all][indices[:aa][1], :accz],
		seriestype=:scatter, 
		markersize = 1,
		markerstrokewidth = 0,
		smooth = true,
		title = "accelerometer xyz-axes",
		titlefontsize = 12,
		xlims = (-5, 5),
		ylims = (-5, 5),
		zlims = (5, 15),
		color = :darktest,
		labels = indices[:aa][2],
		dpi = 200
	)
	
	function acc2plot(idxs, title)
		Plots.plot!(
			data[:all][idxs, :accx], 
			data[:all][idxs, :accy], 
			data[:all][idxs, :accz],
			seriestype=:scatter, 
			markersize = 1,
			markerstrokewidth = 0,
			smooth = true,
			labels = title
		)
	end
	
	acc2plot(indices[:br][1], indices[:br][2])
	acc2plot(indices[:ll][1], indices[:ll][2])
	acc2plot(indices[:rl][1], indices[:rl][2])
	acc2plot(indices[:lt][1], indices[:lt][2])
	acc2plot(indices[:rt][1], indices[:rt][2])
	acc2plot(indices[:na][1], indices[:na][2])
end

# ╔═╡ a2b5e936-72cd-11eb-343b-6d949241a596
begin
	Plots.plot(
		data[:all][indices[:aa][1], :gyrox], 
		data[:all][indices[:aa][1], :gyroy], 
		data[:all][indices[:aa][1], :gyroz],
		seriestype=:scatter, 
		markersize = 1,
		markerstrokewidth = 0,
		smooth = true,
		title = "gyroscope xyz-axes",
		titlefontsize = 12,
		xlims = (-1, 1),
		ylims = (-1, 1),
		zlims = (-1, 1),
		color = :darktest,
		labels = indices[:aa][2],
		dpi = 200
	)
	
	function gyro2plot(idxs, title)
		Plots.plot!(
			data[:all][idxs, :gyrox], 
			data[:all][idxs, :gyroy], 
			data[:all][idxs, :gyroz],
			seriestype=:scatter, 
			markersize = 1,
			markerstrokewidth = 0,
			smooth = true,
			labels = title
		)
	end
	
	gyro2plot(indices[:br][1], indices[:br][2])
	gyro2plot(indices[:ll][1], indices[:ll][2])
	gyro2plot(indices[:rl][1], indices[:rl][2])
	gyro2plot(indices[:lt][1], indices[:lt][2])
	gyro2plot(indices[:rt][1], indices[:rt][2])
	gyro2plot(indices[:na][1], indices[:na][2])
end

# ╔═╡ fa67cfac-72d0-11eb-1198-1d0037923789
begin
	Plots.plot(
		data[:all][indices[:aa][1], :magx], 
		data[:all][indices[:aa][1], :magy], 
		data[:all][indices[:aa][1], :magz],
		seriestype=:scatter, 
		markersize = 1,
		markerstrokewidth = 0,
		smooth = true,
		title = "magnetometer xyz-axes",
		titlefontsize = 12,
		ylims = (0, 1),
		xlims = (-10, 38),
		zlims = (-30, 10),
		color = :darktest,
		labels = indices[:aa][2],
		dpi = 200
	)
	
	function mag2plot(idxs, title)
		Plots.plot!(
			data[:all][idxs, :magy], 
			data[:all][idxs, :magx], 
			data[:all][idxs, :magz],
			seriestype=:scatter, 
			markersize = 1,
			markerstrokewidth = 0,
			smooth = true,
			labels = title
		)
	end
	
	mag2plot(indices[:br][1], indices[:br][2])
	mag2plot(indices[:ll][1], indices[:ll][2])
	mag2plot(indices[:rl][1], indices[:rl][2])
	mag2plot(indices[:lt][1], indices[:lt][2])
	mag2plot(indices[:rt][1], indices[:rt][2])
	mag2plot(indices[:na][1], indices[:na][2])
end

# ╔═╡ 49d629d4-72d2-11eb-1eda-61cf9ca70c29
md"For linear acceleration, accelerometer and gyroscope there are distinguishable trend lines for each event. There are also some visible clusters on 3D plots for these sensors. The 3D plot for magnetometer show that there are circular patches for each event what may indicate that the drivers drove in different directions in relation to the ground. I swapped x-axis with y-axis because changes on x-axis are very small."

# ╔═╡ 9204282a-7339-11eb-28b2-29c1b1d17c87
md"##### Feature Engineering"

# ╔═╡ d7af6286-7339-11eb-0bc3-9f50da3533d1
md"I would like to convert time series data into tabular data to allow the use of standard machine learning algorithms and techniques. To do this I am going to use a technique called Rolling Window Feature. This technique is based on calculating some statistical values based on past values. So time series in a given window is represented by some statistical values like mean, standard deiation and so on. While moving the window, new rows of input vector X are generated."

# ╔═╡ ea8d838c-733a-11eb-0024-2f5e99262151
md"The most important paramterer for sliding window is window size, lets call it `δ` (small delta). It says how many records will be enclosed in one frame. Let's start with `δ ≈ 1.5 s`."

# ╔═╡ 0e2d6c80-7354-11eb-1242-6346140cf353
δ = 1.5

# ╔═╡ 67b45524-733c-11eb-02e1-e3069a955a4e
δf = round(Int, hz * δ)

# ╔═╡ f57e3f18-7342-11eb-1583-4545375a077e
md"δf is a parameter indicating how many datapoints should be used in a given frame."

# ╔═╡ 1cb1bf06-74ba-11eb-1008-bf5a2761d029
md"I remove `non-aggressive_event` class from the processed dataset as we want to detect aggressive events only, and this class is ambiguous. Also removing `aggressive_left_lane_change` classes and `aggressive_right_lane_change` because they have only 4 and 5 complete examples respectively which is too small for reliable training and testing."

# ╔═╡ 8469d244-7507-11eb-071e-8deb1ba015de
md"Downsampling of `none` event is performed to equalize number of samples."

# ╔═╡ 7b3aee26-74ba-11eb-3665-11574c6f247d
begin
	negidxs = 
		(data[:all].event .!= "aggressive_left_lane_change") .& 
		(data[:all].event .!= "aggressive_right_lane_change") .&
		(data[:all].event .!= "non-aggressive_event") .&
		(data[:all].event .!= "none")
	
	nonedf = data[:all][indices[:no][1], :]
	
	data[:filtered] = [data[:all][negidxs, :]; nonedf[1:5000, :]]
	nothing
end

# ╔═╡ b068be44-7342-11eb-209b-35abb1d99a7c
md"Lets write a function that creates frames (described by `δ` parameter) and returns `X` matrix and vector `y` ready for training."

# ╔═╡ af3e9804-7342-11eb-1425-8b21a5a62fbe
function data2frames(df::DataFrame, window::Int, exclude::Vector{Symbol})
	features = df[!, Not(exclude)]
	
	nfun = 6
	nattr = ncol(features) * nfun
	nrows = div(nrow(df), window) + (rem(nrow(df), window) > 0 ? 1 : 0)

	X = zeros(Float32, nrows, nattr)
	
	colnames = String[]
	for name in names(features)
		push!(colnames, string(name) * "_mean")
		push!(colnames, string(name) * "_std")
		push!(colnames, string(name) * "_median")
		push!(colnames, string(name) * "_min")
		push!(colnames, string(name) * "_max")
		push!(colnames, string(name) * "_kurtosis")
	end

	for i in 1:nrows
		rowstart = (i - 1) * window + 1
		rowend = i * window < nrow(df) ? i * window : nrow(df)
		for (coln, col) in enumerate(eachcol(features))
			cn = (coln - 1) * nfun + 1
			println((rowstart, rowend))
			X[i, cn + 0] = mean(col[rowstart:rowend])
			X[i, cn + 1] = std(col[rowstart:rowend])
			X[i, cn + 2] = median(col[rowstart:rowend])
			X[i, cn + 3] = min(col[rowstart:rowend]...)
			X[i, cn + 4] = max(col[rowstart:rowend]...)
			X[i, cn + 5] = kurtosis(col[rowstart:rowend])
		end
	end

	startidx = div(window, 2)
	rlabels = df.event[startidx:window:end]
	diff = length(rlabels) < nrows
	if diff > 0
		push!(rlabels, df.event[end - div(diff, 2)])
	end
	
	y = convertlabel(LabelEnc.Indices{Float32}, rlabels)
	
	X, y, rlabels, colnames
end

# ╔═╡ 59f60efc-737b-11eb-2a85-e703f2cd1c0f
md"This function creates input matrix `X` which for each feature column makes 7 new columns with statistics for each frame: min, max, mean, median, and kurtosis. Number of rows in the matrix `X` is less by `δ - 1`."

# ╔═╡ 234ce1d0-7350-11eb-278c-55cf76406c84
X, y, yl, colnames = data2frames(
	data[:filtered], 
	δf, 
	[:sss, :event, :accx, :accy, :accz]
)

# ╔═╡ f64abf92-7393-11eb-1e7d-d173463e6054
md"I removed accelerometer from input features as it's highly correlated with linear acceleration."

# ╔═╡ 0c76a06e-7476-11eb-21e2-390af3cd6c95
md"Now everything is ready for training."

# ╔═╡ 0d261a86-7394-11eb-2eb2-7bea1c0abc28
md"##### Modeling and results"

# ╔═╡ 2dabbede-742d-11eb-259b-5feae6b5457a
md"Distribution of the classes is unbalanced, so let's see balancing weights."

# ╔═╡ 51c9e7f0-742d-11eb-192d-1b74a13fc1dd
begin
	ylabels = unique(yl)
	clweights = Dict(
		zip(
			ylabels, 
			convert.(Float32, compute_class_weight("balanced", ylabels, yl))
		)
	)
end

# ╔═╡ 0f42ed8a-743c-11eb-1a70-13bce7018a05
md"Create models: random forrest and KNN."

# ╔═╡ 9a347f20-7453-11eb-13d1-993c4b20c595
begin
	rfc = RandomForestClassifier(
		random_state = 42,
		class_weight = "balanced"
	)
	
	knn = KNeighborsClassifier(
		n_neighbors = 3
	)
	
	gnb = GaussianNB()

	nothing
end

# ╔═╡ c642845c-745a-11eb-3872-a181e678adf5
md"50 / 50 train - test split mainly for training and testing execution time comparison."

# ╔═╡ d1831c78-745a-11eb-373d-394225a7bb11
X_train, X_test, y_train, y_test = train_test_split(
	X, 
	y,
	stratify = y,
	test_size = 0.5
)

# ╔═╡ addc7d6a-7506-11eb-0129-6540c34cd7e4
md"Normalization is not helpful in our case."

# ╔═╡ d01409be-7505-11eb-056e-5528166698e7
begin
	Xn = normalize(X)
	Xn_train = normalize(X_train)
	Xn_test = normalize(X_test)
	nothing
end

# ╔═╡ e36f2620-7467-11eb-0f9c-0923a14cd3e8
md"Random Forest 50/50 test."

# ╔═╡ 0cada67a-7397-11eb-3a7c-093c685b6dd5
with_terminal() do
	traintime = @elapsed fit!(rfc, X_train, y_train)
	
	testtime = @elapsed begin
		y_pred = predict(rfc, X_test)
	end
	
	println(classification_report(y_test, y_pred, target_names = ylabels))
	
	rfcaccuracy = sum(y_pred .== y_test) / length(y_test) * 100
	println("accuracy: $rfcaccuracy %")
	println("train time: $traintime seconds")
	println("test time: $testtime seconds")
end

# ╔═╡ 56adc908-745a-11eb-2e8e-0949b76c6bd0
md"KNN 50/50 test."

# ╔═╡ 6ff35a96-73bb-11eb-2609-1b8d06dcd43c
with_terminal() do
	traintime = @elapsed fit!(knn, X_train, y_train)
	
	testtime = @elapsed begin
		y_pred = predict(knn, X_test)
	end
	
	println(classification_report(y_test, y_pred, target_names = ylabels))
	
	rfcaccuracy = sum(y_pred .== y_test) / length(y_test) * 100
	println("accuracy: $rfcaccuracy %")
	println("train time: $traintime seconds")
	println("test time: $testtime seconds")
end

# ╔═╡ 7f68a6e0-74fc-11eb-0921-d55eb3ad472c
md"Gaussian Naive Bayes 50/50 test."

# ╔═╡ 80c451b4-74fd-11eb-30f6-07ba4f758863
with_terminal() do	
	traintime = @elapsed fit!(gnb, X_train, y_train)
	
	testtime = @elapsed begin
		y_pred = predict(gnb, X_test)
	end
	
	println(classification_report(y_test, y_pred, target_names = ylabels))
	
	rfcaccuracy = sum(y_pred .== y_test) / length(y_test) * 100
	println("accuracy: $rfcaccuracy %")
	println("train time: $traintime seconds")
	println("test time: $testtime seconds")
end

# ╔═╡ 0becb3b6-743c-11eb-3dd6-cfbabad66553
md"Stratified cross-validation with 2 folds. I use only 2 folds because there are not many complete events available (mostly 10 - 15 per class) in the dataset."

# ╔═╡ 4ca8814a-7456-11eb-380f-a33c8fa834f5
skf = StratifiedKFold(y, n_folds = 2, shuffle = true, random_state = 42)

# ╔═╡ f3380f10-73a9-11eb-072c-07962bb0d4a3
function classification_report_with_accuracy_score(
	y_true, 
	y_pred;
	target_names = ylabels
)
	println(classification_report(y_true, y_pred, target_names = target_names))
	return accuracy_score(y_true, y_pred)
end

# ╔═╡ 609a2c60-7467-11eb-0cc8-83a9d2ffe790
md"Random forest cross-validation."

# ╔═╡ 98145dbe-743c-11eb-0342-c5a4069c4e36
with_terminal() do	
	rfcaccuracy = 100 * cross_val_score(
		rfc, 
		X, 
		y,
		cv = skf, 
		scoring = make_scorer(
			classification_report_with_accuracy_score, 
			target_names = ylabels
		)
	)
	println("mean accuracy: $(mean(rfcaccuracy)) %")
end

# ╔═╡ 676faae4-7467-11eb-0ede-6d376aa8113b
md"K-Nearest-Neighbors cross-validation."

# ╔═╡ b592d830-73a1-11eb-11ea-891151e7983b
with_terminal() do	
	rfcaccuracy = 100 * cross_val_score(
		knn, 
		X, 
		y,
		cv = skf, 
		scoring = make_scorer(
			classification_report_with_accuracy_score, 
			target_names = ylabels
		)
	)
	println("mean accuracy: $(mean(rfcaccuracy)) %")
end

# ╔═╡ ebc07afc-74fc-11eb-2cea-ad39fbe43470
md"Gaussian Naive Bayes cross-validation."

# ╔═╡ f7407a3a-74fc-11eb-3f52-5165bd7c661b
with_terminal() do	
	rfcaccuracy = 100 * cross_val_score(
		gnb, 
		X, 
		y,
		cv = skf, 
		scoring = make_scorer(
			classification_report_with_accuracy_score, 
			target_names = ylabels
		)
	)
	println("mean accuracy: $(mean(rfcaccuracy)) %")
end

# ╔═╡ d352f308-7398-11eb-2ff1-f3c19646aa00
md"Random forest preformance is the best (accuracy and makro average f1 score about **93%**). However Gaussian Naive Bayes has also a good score and works much faster."

# ╔═╡ bf82022e-7477-11eb-2d64-99228c954026
md"Check feature importance to select unrelevant features, if any."

# ╔═╡ c04c4372-74d9-11eb-34db-6727e942c3d6
begin
	fi = DataFrame(name = colnames, importance = rfc.feature_importances_)
	set_default_plot_size(18cm, 9cm)
	Gadfly.plot(
		fi, 
		x = :name, 
		y = :importance, 
		color = :name, 
		Geom.bar,
		style(key_position = :none),
		Guide.xlabel(""),
		Guide.ylabel(nothing),
		Guide.colorkey(title = "feature importance")
	)
end

# ╔═╡ ac7d8c7e-74d9-11eb-0e87-a395f0b026b0
md"And importance of sensors without transformations and axes."

# ╔═╡ c8fa2f02-7477-11eb-2287-518802122b6a
begin
	rfi = rfc.feature_importances_
	si = DataFrame(
		name = ["linear acceleration", "gyroscope", "magnetometer"],
		importance = Float64[mean(rfi[1:18]), mean(rfi[19:36]), mean(rfi[37:54])]
	)
	set_default_plot_size(8cm, 8cm)
	Gadfly.plot(
		si, 
		x = :name, 
		y = :importance, 
		color = :name, 
		Geom.bar,
		style(key_position = :none),
		Guide.xlabel(""),
		Guide.ylabel(nothing),
		Guide.colorkey(title = "feature importance")
	)
end

# ╔═╡ 3edaf4ee-7479-11eb-11d1-0761f45217a3
md"As I expected in EDA phase `magx` feature is irrelevant for random forrest model. Let's remove it and check scores and execution time for random forest as it is the best classifier."

# ╔═╡ e1a3d224-747c-11eb-3727-c5cfeaa4949f
X2, y2, yl2, colnames2 = data2frames(
	data[:filtered], 
	δf, 
	[:sss, :event, :accx, :accy, :accz, :magx]
)

# ╔═╡ f6e9ad98-74b8-11eb-06b2-5da43d795a54
ylabels2 = unique(yl2)

# ╔═╡ 0a86cbec-747d-11eb-20ef-45e0d2be6651
X2_train, X2_test, y2_train, y2_test = train_test_split(
	X2, 
	y2,
	stratify = y2,
	test_size = 0.5
)

# ╔═╡ 3627e6dc-747d-11eb-30e1-472ffff01a13
md"Random forest cross-validation."

# ╔═╡ 81c28d40-7595-11eb-0569-6b5557dc1410
md"Lest log this experiment to mlflow."

# ╔═╡ 99390580-7595-11eb-123f-b31bf9a83930
begin
	params  = Dict(
		"model" => "random forest",
		"nfolds" => 2,
		"random_state" => 42,
		"class_weight" => "balanced",
		"window" => δ
	)
	
	mlflow.log_param("features", colnames2)
	mlflow.log_param("params", params)
	mlflow.log_param("classes", ylabels2)
end

# ╔═╡ 455d52e0-747d-11eb-1f0a-23d661bdef04
with_terminal() do
	rfcaccuracy = 100 * cross_val_score(
		rfc, 
		X2, 
		y2,
		cv = skf, 
		scoring = make_scorer(
			classification_report_with_accuracy_score, 
			target_names = ylabels
		)
	)

	println("mean accuracy: $(mean(rfcaccuracy)) %")
	mlflow.log_metric("cv_mean_accuracy", mean(rfcaccuracy))
end

# ╔═╡ 2722e97e-747e-11eb-27f8-15104e9564ed
md"Execution time is nearly the same but the score a bit better. Model is simpler now (40 features) so it is a good decision to remove this feature."

# ╔═╡ 8a11530e-747e-11eb-30f2-318bf235d845
md"Confusion matrix for random forest."

# ╔═╡ 111432ac-7484-11eb-1d67-532a9a95ec82
begin
	fit!(rfc, X2_train, y2_train)
	
	y2_pred_cmrf = predict(rfc, X2_test)
	 
	shortylabels = String[]
	for label in ylabels
		splitted = split(label, "_")
		if length(splitted) > 1
			push!(shortylabels, join(splitted[2:end], "_"))
		else
			push!(shortylabels, label)
		end
	end
	
	confmatrix_rf = hcat(
		DataFrame(class = reverse(shortylabels)),
		DataFrame(confusion_matrix(y2_test, y2_pred_cmrf))
	)
	
	cmnames = vcat("class", reverse(shortylabels))
	rename!(confmatrix_rf, Symbol.(cmnames))
	confmatrix_rf
end

# ╔═╡ 5d575f82-7483-11eb-1541-590a6be93c4b
md"Confusion matrix shows that the model makes only a few mistakes (but the number of complete events is small). The most common error is between aggressive right turn and acceleration."

# ╔═╡ cdd2d530-7597-11eb-0ccf-f99f104849ba
mlflow.end_run()

# ╔═╡ Cell order:
# ╟─644a7e0a-6eb8-11eb-261c-9feb1fe2b353
# ╟─95d96c92-6ebd-11eb-136d-1d3a3f7f0893
# ╟─b2283ab8-6ebd-11eb-31e3-5770050b62b6
# ╠═ed55f87e-6ebc-11eb-0e31-539f6c4616cc
# ╟─7b591c3a-6ebf-11eb-1824-27b2057eda6e
# ╠═d3df7b06-6eba-11eb-0985-e9e458ee8dc8
# ╟─d989fcaa-7434-11eb-3223-0935342ccefd
# ╠═9185384e-7393-11eb-2be3-dba24541dfbf
# ╟─5d96ac50-7593-11eb-3686-b9be36643e28
# ╠═54a25860-7593-11eb-0dc8-c189891997f7
# ╟─c71223e6-715c-11eb-3aa0-49eef018dde0
# ╠═f79fe3b0-715c-11eb-0517-4ff69996aca5
# ╟─cb9cc790-6ec0-11eb-1163-8301e347a97f
# ╟─7686accc-7027-11eb-0917-1db37660474c
# ╟─61fe483e-7034-11eb-13fa-3b6e28ce2c30
# ╠═157566bc-7022-11eb-3d43-697e6d009061
# ╟─11132b80-703a-11eb-1b79-d376afe6099e
# ╠═e1bbb95a-6ec0-11eb-2c5a-917b55d9ebae
# ╟─b1ecfbde-7209-11eb-2d7b-79a50fe9ae91
# ╟─c149e238-704a-11eb-3b34-e165bb13e4cc
# ╠═9d22af32-7060-11eb-073f-4debb0fd8597
# ╟─6f5c89ca-705f-11eb-3028-7d8f072b8e32
# ╟─8586aeb8-705f-11eb-1620-2df3b3002608
# ╠═7b5c15ec-704b-11eb-1dbe-373102452d5b
# ╟─a00fd488-7081-11eb-1e23-e1d1011d9c8a
# ╠═81ba6b40-70ec-11eb-2369-c76fb4390f60
# ╠═e9eaacda-70e6-11eb-04f4-69c351839996
# ╟─7ac967d2-70e7-11eb-1af1-a9d8e82c80fc
# ╟─eb944420-7063-11eb-3188-533bd3f2d3ed
# ╠═50b5980e-70ff-11eb-3db0-67d1bc0d4361
# ╟─cc76eb6e-70ff-11eb-26f7-89e592f0187c
# ╠═18e7214e-7083-11eb-2dab-2b40ff9eee3d
# ╟─fea57d42-70eb-11eb-1249-8bd9af09bfcb
# ╟─2cd75a90-70fe-11eb-314f-a9b054936350
# ╠═4896f82e-7065-11eb-220e-917923327763
# ╟─b5f31a70-70f7-11eb-3060-e5ec5cc2ae6b
# ╠═42f072b6-70f0-11eb-1d11-19c6d66e286c
# ╟─1a9bae16-70f0-11eb-1bd0-dfeb3e8adb08
# ╠═280c9254-70fa-11eb-30d9-c57870b21c0a
# ╟─b6b5ae14-7109-11eb-225e-b34060198e55
# ╟─4bf118b0-7119-11eb-10fc-8390fa2b49a3
# ╠═59a62a10-7144-11eb-2da1-ed4d5b9dce07
# ╟─12ac1eac-7148-11eb-0c8d-1de16d04534d
# ╟─e60ec0ea-711a-11eb-29cf-b91ecdb47d87
# ╠═95372af0-710a-11eb-0468-759d65813306
# ╟─03fb8700-7125-11eb-04eb-61e2e3213b30
# ╠═e81d7a0c-7124-11eb-32fa-c7d4611e9fa4
# ╟─20fa98e2-7188-11eb-0ec9-aba67380afa8
# ╠═8f6e7c22-7187-11eb-23d3-157713f21e0b
# ╟─f8a9db0e-711a-11eb-0013-0fb1ff1a02e1
# ╠═19bc4e3a-711b-11eb-0fb9-915e415d545d
# ╟─dda67424-7148-11eb-00a8-75807d6c825d
# ╠═42780f5c-7149-11eb-3461-bb606acb1d67
# ╟─012af6d4-714b-11eb-1b7f-e92ef44b646d
# ╟─3cfc5fbc-7151-11eb-3ea0-f558fb038e29
# ╠═61bfcc94-7151-11eb-2534-d94a806213c9
# ╟─3cbb0d64-7156-11eb-3c81-1f9c6044ccb8
# ╟─9ecf1b06-727f-11eb-3d99-310b3ff751d3
# ╟─de974a20-7150-11eb-05b4-497bcf6c345c
# ╠═f17799c6-7150-11eb-17c5-a5ea424cbb39
# ╟─07009f06-71fe-11eb-24b4-134600e50d86
# ╟─46aa540e-751d-11eb-1b77-23f3adc6c7fd
# ╟─2f4095de-7159-11eb-2945-254c26081516
# ╟─26d7e2ca-715a-11eb-0b75-458e78fec242
# ╟─c0be143c-7172-11eb-1f87-55db449dc586
# ╟─97c57b7c-7188-11eb-0619-8d900fbe23bc
# ╟─78f9b418-72ac-11eb-2822-17db36df1c60
# ╟─12a8e618-72be-11eb-1b73-bb0bcc914d37
# ╟─1144b95c-7190-11eb-161e-ab808884e4a4
# ╟─3cf876a4-727d-11eb-248c-db2f5a822c21
# ╠═02a4bef6-72cd-11eb-2387-2783e5833851
# ╟─f641dc68-72c0-11eb-1eca-fd4af026a286
# ╟─66c1445c-72d2-11eb-2a05-3fdb04a1d92b
# ╟─a2b5e936-72cd-11eb-343b-6d949241a596
# ╟─fa67cfac-72d0-11eb-1198-1d0037923789
# ╟─49d629d4-72d2-11eb-1eda-61cf9ca70c29
# ╟─9204282a-7339-11eb-28b2-29c1b1d17c87
# ╟─d7af6286-7339-11eb-0bc3-9f50da3533d1
# ╟─ea8d838c-733a-11eb-0024-2f5e99262151
# ╠═0e2d6c80-7354-11eb-1242-6346140cf353
# ╠═67b45524-733c-11eb-02e1-e3069a955a4e
# ╟─f57e3f18-7342-11eb-1583-4545375a077e
# ╟─1cb1bf06-74ba-11eb-1008-bf5a2761d029
# ╟─8469d244-7507-11eb-071e-8deb1ba015de
# ╠═7b3aee26-74ba-11eb-3665-11574c6f247d
# ╟─b068be44-7342-11eb-209b-35abb1d99a7c
# ╠═af3e9804-7342-11eb-1425-8b21a5a62fbe
# ╟─59f60efc-737b-11eb-2a85-e703f2cd1c0f
# ╟─234ce1d0-7350-11eb-278c-55cf76406c84
# ╟─f64abf92-7393-11eb-1e7d-d173463e6054
# ╟─0c76a06e-7476-11eb-21e2-390af3cd6c95
# ╟─0d261a86-7394-11eb-2eb2-7bea1c0abc28
# ╟─2dabbede-742d-11eb-259b-5feae6b5457a
# ╠═51c9e7f0-742d-11eb-192d-1b74a13fc1dd
# ╟─0f42ed8a-743c-11eb-1a70-13bce7018a05
# ╠═9a347f20-7453-11eb-13d1-993c4b20c595
# ╟─c642845c-745a-11eb-3872-a181e678adf5
# ╠═d1831c78-745a-11eb-373d-394225a7bb11
# ╟─addc7d6a-7506-11eb-0129-6540c34cd7e4
# ╠═d01409be-7505-11eb-056e-5528166698e7
# ╟─e36f2620-7467-11eb-0f9c-0923a14cd3e8
# ╠═0cada67a-7397-11eb-3a7c-093c685b6dd5
# ╟─56adc908-745a-11eb-2e8e-0949b76c6bd0
# ╠═6ff35a96-73bb-11eb-2609-1b8d06dcd43c
# ╟─7f68a6e0-74fc-11eb-0921-d55eb3ad472c
# ╠═80c451b4-74fd-11eb-30f6-07ba4f758863
# ╟─0becb3b6-743c-11eb-3dd6-cfbabad66553
# ╠═4ca8814a-7456-11eb-380f-a33c8fa834f5
# ╠═f3380f10-73a9-11eb-072c-07962bb0d4a3
# ╟─609a2c60-7467-11eb-0cc8-83a9d2ffe790
# ╠═98145dbe-743c-11eb-0342-c5a4069c4e36
# ╟─676faae4-7467-11eb-0ede-6d376aa8113b
# ╠═b592d830-73a1-11eb-11ea-891151e7983b
# ╟─ebc07afc-74fc-11eb-2cea-ad39fbe43470
# ╠═f7407a3a-74fc-11eb-3f52-5165bd7c661b
# ╟─d352f308-7398-11eb-2ff1-f3c19646aa00
# ╟─bf82022e-7477-11eb-2d64-99228c954026
# ╟─c04c4372-74d9-11eb-34db-6727e942c3d6
# ╟─ac7d8c7e-74d9-11eb-0e87-a395f0b026b0
# ╠═c8fa2f02-7477-11eb-2287-518802122b6a
# ╟─3edaf4ee-7479-11eb-11d1-0761f45217a3
# ╠═e1a3d224-747c-11eb-3727-c5cfeaa4949f
# ╠═f6e9ad98-74b8-11eb-06b2-5da43d795a54
# ╠═0a86cbec-747d-11eb-20ef-45e0d2be6651
# ╟─3627e6dc-747d-11eb-30e1-472ffff01a13
# ╟─81c28d40-7595-11eb-0569-6b5557dc1410
# ╠═99390580-7595-11eb-123f-b31bf9a83930
# ╠═455d52e0-747d-11eb-1f0a-23d661bdef04
# ╟─2722e97e-747e-11eb-27f8-15104e9564ed
# ╟─8a11530e-747e-11eb-30f2-318bf235d845
# ╠═111432ac-7484-11eb-1d67-532a9a95ec82
# ╟─5d575f82-7483-11eb-1541-590a6be93c4b
# ╠═cdd2d530-7597-11eb-0ccf-f99f104849ba
