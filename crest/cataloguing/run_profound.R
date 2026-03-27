library(hash)
library(glue)
`%notin%` <- Negate(`%in%`)

# Function to parse command-line arguments.
parse_args <- function(args) {

  arg_list <- list()

  # Split each argument on '='.
  for (arg in args) {

    parts <- strsplit(arg, "=")[[1]]

    # Ensure there are exactly two parts.
    if (length(parts) == 2) {
      arg_name <- parts[1]
      arg_value <- parts[2]

      # Store and convert to numeric if possible.
      numeric_value <- suppressWarnings(as.numeric(arg_value))

      if (!is.na(numeric_value)) {
        arg_list[[arg_name]] <- numeric_value
      } else {
        arg_list[[arg_name]] <- arg_value
      }
    } else {
      warning(paste("Invalid argument format:", arg))
    }
  }
  return(arg_list)
}

# Get the command-line arguments.
args <- commandArgs(trailingOnly = TRUE)
arg_list <- parse_args(args)

# Resolve this script's directory robustly under Rscript.
script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
if (length(file_arg) == 0) {
  stop("Could not determine run_profound.R path from commandArgs().")
}
script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
crest_path <- dirname(script_path)

# Get the ProFound class.
pf_name <- file.path(crest_path, "profound.R")
source(pf_name)

# Initalise the profound object using the config path.
p_run <- ProFound(arg_list[["config_path"]])

# Are we extracting?
if (arg_list[["type"]] == "extract") {

  # Get the image paths in the correct format.
  if ("img2" %in% names(arg_list)) {
    science <- list(arg_list[["img1"]], arg_list[["img2"]])
  } else {
    science <- arg_list[["img1"]]
  }

  # Get the requested outputs.
  if (arg_list[["outputs"]] == "None") {
    outputs <- NULL
  } else {
    outputs <- strsplit(arg_list[["outputs"]], ",")[[1]]
  }

  # Pass additional parameters.
  parameters <- hash()
  for (key in names(arg_list)) {
    if (key %notin% c("img1", "img2", "outputs", "cat_name", "outdir", "config_path", 
                      "type")) {
      parameters[[key]] <- arg_list[[key]]
    }
  }

  # Finally, run ProFound.
  out_name <- p_run$extract(science, parameters, outputs, arg_list[["cat_name"]], arg_list[["outdir"]])

  # Print the output for Python to capture.
  cat("out_name: ", out_name, "\n")

  # Or measuring the depth?
} else if (arg_list[["type"]] == "depth") {

  # Pass additional parameters.
  parameters <- hash()
  for (key in names(arg_list)) {
    if (key %notin% c("img1", "config_path", "type", "psf", "radius",
                      "max_apers", "max_iters", "mask", "error")) {
      parameters[[key]] <- arg_list[[key]]
    }
  }

  # Run ProFound.
  depth <- p_run$measure_depth(arg_list[["img1"]], arg_list[["psf"]], arg_list[["mask"]],
                               arg_list[["error"]], parameters, arg_list[["radius"]],
                               arg_list[["max_apers"]], arg_list[["max_iters"]])

  # Print the output for Python to capture.
  cat("Depth: ", depth, "\n")

}