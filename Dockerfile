FROM julia:latest

# user
# RUN useradd --create-home --shell /bin/bash drivingevents

# app
RUN mkdir /drivingevents
COPY . /drivingevents
WORKDIR /drivingevents

# RUN chown drivingevents:drivingevents -R *
# RUN chown drivingevents:drivingevents -R /drivingevents

RUN chmod +x bin/repl
RUN chmod +x bin/server
RUN chmod +x bin/runtask

# USER drivingevents

RUN julia -e "using Pkg; Pkg.activate(\".\"); Pkg.instantiate(); Pkg.precompile(); "

# ports
EXPOSE 5858

ENV JULIA_DEPOT_PATH "~/.julia"
ENV GENIE_ENV "dev"
ENV HOST "0.0.0.0"
ENV PORT "5858"

CMD ["bin/server"]