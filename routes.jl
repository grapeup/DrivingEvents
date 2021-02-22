using Genie.Router, Genie.Requests, Genie.Responses, Genie.Exceptions
using Genie.Renderer.Json

route("/notebook") do
	serve_static_file("drivingevents.html")
end
