using CSV
using DataFrames
using Dates
using PyPlot

function _parse_datetime_input(x)
    if x isa DateTime
        return x
    elseif x isa Date
        return DateTime(x)
    elseif x isa AbstractString
        return try
            DateTime(x, dateformat"dd-u-yyyy HH:MM:SS")
        catch
            try
                DateTime(x, dateformat"yyyy-mm-dd HH:MM:SS")
            catch
                DateTime(x, dateformat"yyyy-mm-dd")
            end
        end
    else
        error("Unsupported date input: $x")
    end
end

function plot_load_and_pv(
    start_date,
    end_date;
    data_path::AbstractString=joinpath(@__DIR__, "Processed_Data_15min.csv"),
)
    df = CSV.read(data_path, DataFrame)
    df.DateTime = DateTime.(df.DateTime, dateformat"dd-u-yyyy HH:MM:SS")

    t0 = _parse_datetime_input(start_date)
    t1 = _parse_datetime_input(end_date)
    if t1 < t0
        error("`end_date` must be greater than or equal to `start_date`.")
    end

    filtered = df[(df.DateTime .>= t0) .& (df.DateTime .<= t1), :]
    nrow(filtered) > 0 || error("No data found between $t0 and $t1.")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(filtered.DateTime, filtered.Load, linewidth=2, color="tab:blue", label="Load")
    ax.plot(filtered.DateTime, filtered.PVPower, linewidth=2, color="tab:orange", label="PV")
    ax.set_title("Load and PV Power")
    ax.set_xlabel("Date")
    ax.set_ylabel("Power")
    ax.grid(true, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    display(fig)

    return fig, ax, filtered
end

plot_load_and_pv("2017-05-01", "2017-05-30")
