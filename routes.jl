using Genie.Router, Genie.Requests, Genie.Responses, Genie.Exceptions
using Genie.Renderer.Json

route("/notebook") do
	serve_static_file("drivingevents.html")
end

route("/train") do
	serve_static_file("drivingevents.html")
end

route("/predict") do
	serve_static_file("drivingevents.html")
end