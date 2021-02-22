module DrivingEvents

using Genie, Logging, LoggingExtras

function main()
    Base.eval(Main, :(const UserApp = DrivingEvents))

    Genie.genie(; context = @__MODULE__)

    Base.eval(Main, :(const Genie = DrivingEvents.Genie))
    Base.eval(Main, :(using Genie))
end

end
