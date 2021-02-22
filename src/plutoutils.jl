using PlutoUI
using DataFrames

function head(df::DataFrame) 
    PlutoUI.WithIOContext(df, displaysize=(3,100))
end
