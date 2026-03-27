# Import the relevent libraries.
library(ProFound)
library(Rfits)
library(Rwcs)
library(hash)
library(yaml)
library(glue)
library(rhdf5)
library(stringr)
library(EBImage)

# Import utility functions.
script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
  script_dir <- dirname(script_path)
} else {
  stop("Could not determine profound.R path from commandArgs().")
}
source(file.path(script_dir, "utils.R"))

# This is the main ProFound class.
# Written to be useable directly from R if required.
ProFound <- setRefClass("profound", fields = list(config_path = "character", config = "list",
                                                  verbose = "logical"),

                        # Class for running ProFound in single or dual
                        # image mode uisng a config file and producing
                        # an hdf5 catalogue.

                        methods = list(initialize = function(config_path, verbose = TRUE) {

                          # initialize method for ProFound.

                          # Arguments
                          # ---------
                          # config_path (str)
                          #   Path to the YAML configuration file.
                          # verbose (bool)
                          #   If TRUE, print progress messages.

                          # Store the config filepath and content.
                          .self$config_path <- config_path
                          .self$config <- yaml.load_file(config_path)
                          .self$verbose <- verbose
                        },

                        vmessage = function(...) {

                          # Print a message only when verbosity is
                          # enabled.

                          if (isTRUE(.self$verbose)) {
                            message(...)
                          }
                        },

                        expand_env_var = function(string) {

                          # Exapand environment variables in a string.

                          # Arguments
                          # ---------
                          # string (str)
                          #   The string containing environment
                          #   variables.

                          # Returns
                          # -------
                          # result (str)
                          #   The expanded string.

                          pattern <- "\\$\\{?([A-Za-z_][A-Za-z0-9_]*)\\}?"

                          result <- str_replace_all(string, pattern, function(match) {
                            var_name <- gsub("\\$\\{?([A-Za-z_][A-Za-z0-9_]*)\\}?", "\\1", match)
                            Sys.getenv(var_name, unset = "")
                          })

                          return(result)
                        },

                        update_config = function(parameters) {

                          # Create a new config file with parameters
                          # updated with those provided at runtime.

                          # Arguments
                          # ---------
                          # parameters (hash)
                          #   Key-value pairs of parameters to update.

                          # Returns
                          # -------
                          # new_config (list)
                          #   The updated config file.
                          # att_config (list)
                          #   The updated config file with values able
                          #   to be saved to a hdf5 file.

                          # Make a local copy of the config.
                          new_config <- lapply(.self$config, identity)

                          # Update the config with given parameters.
                          if (length(parameters) > 0) {
                            for (key in names(parameters)) {
                              new_config[[key]] <- parameters[[key]]
                            }
                          }

                          # If "iters" not provided, copy "iters_det".
                          # The former is needed for single image mode.
                          if (!("iters" %in% names(new_config))) {
                            new_config[["iters"]] <- new_config[["iters_det"]]
                          }

                          # Store the config as is for saving as hdf5
                          # attributes.
                          att_config <- lapply(new_config, identity)

                          # Adjust the parameters as needed and expand
                          # environment variables.
                          for (key in names(new_config)) {
                            if (typeof(new_config[[key]]) == "character") {
                              if (new_config[[key]] == "Inf") {
                                new_config[[key]] <- Inf
                              } else if (new_config[[key]] == "None") {
                                new_config[[key]] <- NULL
                              } else {
                                new_config[[key]] <- .self$expand_env_var(new_config[[key]])
                              }
                            }
                          }

                          return(list(new_config = new_config, att_config = att_config))
                        },

                        close_groups = function(file, group, group_name) {

                          # Close a hdf5 group and delete it if empty.

                          # Arguments
                          # ---------
                          # file (H5I_FILE)
                          #   The hdf5 catalogue file object.
                          # group (H5I_GROUP)
                          #   The hdf5 group object to be closed.
                          # group_name (str)
                          #   Name of the group to be closed.

                          if (length(h5ls(group)) == 0) {
                            H5Gclose(group)
                            h5delete(file, group_name)
                          } else {
                            H5Gclose(group)
                          }
                        },

                        measure_depth = function(science, psf, mask = NULL, error = NULL,
                                                 parameters = NULL, radius = 3.33, max_apers = 50,
                                                 max_iters = 50000) {

                          # Measure the 5-sigma point source depth of an
                          # image using random circular apertures.

                          # Arguments
                          # ---------
                          # science (string)
                          #   Path to the fits image to measure from.
                          # psf (string)
                          #   Path to the fits image containing the PSF
                          #   of the science image.
                          # mask (string/NULL)
                          #   Path to fits source mask for the science
                          #   image. If NULL, compute internally.
                          # error (string/NULL)
                          #   Path to fits error map of the science
                          #   image. Only used for coverage masking.
                          # parameters (hash)
                          #   Key-value pairs with which to update the
                          #   config file.
                          # radius (float)
                          #   The radius in pixels of the circular
                          #   apertures to place.
                          # max_apers (int)
                          #   The maximum number of random apertures to
                          #   place.
                          # max_iters (int)
                          #   Maximum attempts at placing a
                          #   non-overlapping aperture.

                          # Returns
                          # -------
                          # depth (float)
                          #   The 5-sigma depth of the image.

                          if (is.null(parameters)) {
                            parameters <- hash()
                          }

                          # Update the config file.
                          configs <- .self$update_config(parameters)
                          img_config <- configs[["new_config"]]
                          att_config <- configs[["att_config"]]

                          # Get the science image.
                          sci <- Rfits_point(science)
                          sci_ <- Rfits_read_image(science)$imDat
                          hdr <- sci$keyvalues
                          sci_dim <- dim(sci)

                          # If user provides a mask via the config, it
                          # can be used.
                          if (typeof(img_config[["mask"]]) == "character") {
                            in_mask <- Rfits_read_image(img_config[["mask"]])
                            img_config[["mask"]] <- in_mask$imDat
                          } else {
                            in_mask <- ifelse(sci_ == img_config[["mask"]], 1, 0)
                          }

                          # This PSF will only be used when generating
                          # a source mask or RMS map.
                          if (!is.null(img_config[["psf"]])) {
                            psf <- Rfits_read_image(img_config[["psf"]])
                            img_config[["psf"]] <- psf$imDat
                            rm(psf)
                          }

                          # Segmentation map cannot be used.
                          img_config[["segim"]] <- NULL

                          # If no mask or RMS provided, use those
                          # generated by ProFound.
                          if (is.null(mask)) {

                            # We only want to produce a segmentation map.
                            manual <- list(image = sci, keepim = FALSE, plot = FALSE, stats = FALSE,
                                           rotstats = FALSE, boundstats = FALSE, nearstats = FALSE,
                                           groupstats = FALSE, group = NULL, haralickstats = FALSE,
                                           keepsegims = TRUE, doChiSq = FALSE)
                            all_params <- modifyList(img_config, manual)

                            profound_run <- do.call(profoundProFound, all_params)
                          }

                          # Assign the different mask components.
                          if (is.null(mask)) {
                            source_mask <- profound_run$segim
                          } else {
                            source_mask <- Rfits_read_image(mask)$imDat
                          }

                          # Identify masked areas.
                          binary_mask <- ifelse(is.na(sci_) | source_mask != 0 | in_mask == 1, 1, 0)

                          if (!is.null(error)) {
                            err <- Rfits_read_image(error)$imDat
                            binary_mask <- ifelse(binary_mask == 1 | err <= 0 | is.na(err), 1, 0)
                            rm(err)
                          }

                          # Dilate mask by the aperture size.
                          dilated_mask <- dilate(binary_mask, makeBrush((2 * radius) + 1, shape = "disc"))
                          valid_positions <- which(dilated_mask == 0, arr.ind = TRUE)

                          rm(sci_)
                          rm(in_mask)
                          rm(source_mask)

                          # Will store aperture locations here
                          apertures <- matrix(NA, nrow = max_apers, ncol = 2)
                          # and place them here.
                          det <- matrix(0, nrow = sci_dim[1], ncol = sci_dim[2])

                          # Place the apertures.
                          i <- 1
                          iters <- 0
                          while (i <= max_apers) {

                            # Get a random location.
                            idx <- sample(nrow(valid_positions), 1)
                            x <- valid_positions[idx, 1]
                            y <- valid_positions[idx, 2]

                            # Place aperture if it doesn't overlap.
                            if (check_overlap(x, y, apertures, radius)) {
                              apertures[i, ] <- c(x, y)
                              det[x, y] <- i

                              i <- i + 1
                              iters <- 0

                              # Try another location if it does.
                            } else {
                              iters <- iters + 1
                            }

                            # Break if maximum attempts are reached.
                            if (iters >= max_iters) {
                              break
                            }
                          }
                          .self$vmessage(glue("Placed {i-1} apertures."))

                          # Even with an aperture, ProFound will only
                          # consider pixels in the segmentation map.
                          det <- dilate(det, makeBrush(size = 2 * radius + 1, shape = "disc"))

                          # Aperture diameter in arcseconds.
                          diam <- 2 * radius * img_config[["pixscale"]]

                          # Now run ProFound, using det as a segmentation
                          # map. Ensure no dilation occurs.
                          manual <- list(image = sci, segim = det, static_photom = TRUE, app_diam = diam,
                                         stats = TRUE, keepim = FALSE, plot = FALSE, sky = 0, redosky = FALSE,
                                         doclip = FALSE, shiftloc = FALSE, redosegim = FALSE, smooth = FALSE,
                                         SBlim = NULL, SBdilate = 0, deblend = FALSE, psf = NULL,
                                         doChiSq = FALSE, rotstats = FALSE, boundstats = FALSE, nearstats = FALSE,
                                         groupstats = FALSE, group = NULL, haralickstats = FALSE,
                                         keepsegims = FALSE)
                          all_params <- modifyList(img_config, manual)

                          profound_run <- do.call(profoundProFound, all_params)

                          # Calculate the MAD of the aperture fluxes.
                          ap_fluxes <- profound_run$segstats[["flux_app"]] * img_config[["flux_conversion"]]
                          mad_ <- mad(ap_fluxes)

                          # Correct using the fraction of the PSF
                          # enclosed by the aperture.
                          psf_ <- Rfits_read_image(psf)$imDat
                          radii <- seq(0.1, nrow(psf_), by = 1)
                          result <- measure_curve_of_growth(psf_, radii)
                          f <- function(r) approx(result[["radii"]], result[["cog"]], xout = r, rule = 2)$y

                          depth <- (5 * mad_) / f(radius)

                          return(depth)

                        },

                        extract = function(science, parameters = NULL, outputs = NULL,
                                           cat_name = NULL, outdir = "./") {

                          # Main extraction method for ProFound.

                          # Arguments
                          # ---------
                          # science (str/List[str])
                          #   If string, path to single image from which
                          #   to detect sources. If List[str], path to
                          #   detection image as the first entry and
                          #   measurement as the second.
                          # parameters (hash)
                          #   Key-value pairs overwritting parameters in
                          #   the config file just for this run.
                          # outputs (NULL/List[str])
                          #   The source extraction outputs to save.
                          #   If NULL, save all available.
                          # cat_name (NULL/str)
                          #   The basename of the catalogue to save.
                          #   If NULL, use the image name.
                          # outdir (str)
                          #   The directory in which to save outputs.

                          # Returns
                          # -------
                          # cat_name (str)
                          #   The full path to the hdf5 catalogue.

                          if (is.null(parameters)) {
                            parameters <- hash()
                          }

                          configs <- .self$update_config(parameters)
                          img_config <- configs[["new_config"]]
                          att_config <- configs[["att_config"]]

                          # Load segmap, mask and PSF if provided.
                          if (!is.null(img_config[["segim"]])) {
                            segim <- Rfits_read_image(img_config[["segim"]])
                            img_config[["segim"]] <- segim$imDat
                            rm(segim)
                          }
                          if (typeof(img_config[["mask"]]) == "character") {
                            mask <- Rfits_read_image(img_config[["mask"]])
                            img_config[["mask"]] <- mask$imDat
                            rm(mask)
                          }
                          if (!is.null(img_config[["psf"]])) {
                            psf <- Rfits_read_image(img_config[["psf"]])
                            img_config[["psf"]] <- psf$imDat
                            rm(psf)
                          }

                          # Are we in single or dual image mode?
                          single <- TRUE
                          if (length(science) == 1) {
                            .self$vmessage("Running in single image mode.")
                            sci <- Rfits_point(science)
                            hdr <- sci$keyvalues
                            temp_name <- glue("{outdir}/{gsub('.fits', '_profound.hdf5', basename(science))}")

                          } else if (length(science) == 2) {
                            .self$vmessage("Running in double image mode.")
                            single <- FALSE
                            det <- Rfits_point(unlist(science[1]))
                            sci <- Rfits_point(unlist(science[2]))
                            hdr <- sci$keyvalues
                            input <- list(det, sci)
                            temp_name <- glue("{outdir}/{gsub('.fits', '_profound.hdf5', basename(unlist(science[2])))}")

                            # Else, raise an error.
                          } else {
                            stop("Image inputs are not the correct format. Use strings for single 
                            image mode and lists of the form [detection, measurement] for double.")
                          }

                          # Set the catalogue name.
                          if (is.null(cat_name)) {
                            cat_name <- temp_name
                          } else {
                            cat_name <- glue("{outdir}/{cat_name}.hdf5")
                          }

                          # Run ProFound.
                          if (single) {
                            img_config[["iters"]] <- img_config[["iters_det"]]
                            manual <- list(image = sci, keepim = FALSE)
                            all_params <- modifyList(img_config, manual)

                            profound_run <- do.call(profoundProFound, all_params)

                          } else {
                            manual <- list(inputlist = input, detectbands = "det", multibands = c("det", "sci"),
                                           keepim = FALSE, totappend = "_total",
                                           colappend = "_colour", grpappend = "_group")
                            all_params <- modifyList(img_config, manual)

                            profound_run <- do.call(profoundMultiBand, all_params)
                          }

                          # Create or replace the hdf5 catalogue.
                          if (file.exists(cat_name)) {
                            file.remove(cat_name)
                          }
                          h5createFile(cat_name)

                          # Load the catalogue and add groups.
                          catalogue <- H5Fopen(cat_name)

                          h5createGroup(catalogue, "photometry")
                          photometry <- H5Gopen(catalogue, "photometry")
                          h5createGroup(catalogue, "groups")
                          groups <- H5Gopen(catalogue, "groups")

                          # Iterate through all available data products
                          # from both types of run.
                          products <- list(profound_run$segstats, profound_run$near, profound_run$group,
                                           profound_run$groupstats, profound_run$cat_tot, profound_run$cat_col,
                                           profound_run$cat_grp)

                          for (data in products) {

                            # Skip if not available.
                            if (!is.null(data)) {

                              # Save group info to group group...
                              current_group <- photometry
                              if (identical(data, profound_run$group) || identical(data, profound_run$groupstats) ||
                                    identical(data, profound_run$cat_grp)) {
                                current_group <- groups
                              }

                              # In double image mode we only want the
                              # science data.
                              if (!single) {
                                sci_columns <- grep("_sci_", names(data), value = TRUE)
                                data <- data[sci_columns]
                                names(data) <- gsub("_sci_", "_", names(data))
                              }

                              # Remove any columns that are all NAs.
                              all_na_columns <- sapply(data, function(x) all(is.na(x)))
                              data <- data[!all_na_columns]

                              # Get list of columns containing fluxes.
                              columns <- list("flux", "sky_mean", "sky_sum", "skyRMS_mean", "skyseg_mean")
                              flux_cols <- list()
                              for (column in columns) {
                                flux_cols <- c(flux_cols, grep(column, names(data), value = TRUE))
                              }
                              # Remove those that aren't fluxes.
                              flux_cols <- setdiff(flux_cols, c("flux_segfrac"))
                              flux_cols <- unlist(flux_cols)

                              # Apply requested flux conversion.
                              for (column in flux_cols) {
                                if (is.numeric(data[[column]])) {
                                  data[[column]] <- data[[column]] * img_config[["flux_conversion"]]
                                }
                              }

                              # If no outputs requested, add all.
                              if (is.null(outputs)) {
                                outputs_ <- names(data)
                              } else {
                                outputs_ <- outputs
                              }
                              outputs_ <- intersect(outputs_, names(data))

                              # Create datasets.
                              for (col in outputs_) {

                                # Some data is in a format that cannot be
                                # saved to hdf5. Save as string that can
                                # be decoded.
                                if (is.recursive(data[[col]]) && any(sapply(data[[col]], is.list))) {
                                  data_to_save <- sapply(data[[col]], function(x) paste(x, collapse = ","))
                                } else {
                                  data_to_save <- unlist(data[[col]])
                                }
                                h5write(data_to_save, file = current_group, name = col)
                              }
                            }
                          }

                          # Add parameter values to catalogue as
                          # attributes.
                          for (key in names(att_config)) {
                            if ((typeof(att_config[[key]]) == "logical") || is.null(att_config[[key]])) {
                              att_config[[key]] <- toString(att_config[[key]])
                            }
                            # Write the attribute to the HDF5 group.
                            h5writeAttribute(attr = att_config[[key]], h5obj = photometry, name = key)
                            h5writeAttribute(attr = att_config[[key]], h5obj = groups, name = key)
                          }

                          # Add provenance metadata used by Python utilities.
                          h5writeAttribute(attr = "ProFound", h5obj = photometry, name = "CODE")
                          h5writeAttribute(attr = as.character(packageVersion("ProFound")),
                                           h5obj = photometry, name = "VERSION")
                          h5writeAttribute(attr = "ProFound", h5obj = groups, name = "CODE")
                          h5writeAttribute(attr = as.character(packageVersion("ProFound")),
                                           h5obj = groups, name = "VERSION")

                          # Function to save each image to its
                          # corresponding filename.
                          save_image <- function(image_, filename_) {
                            Rfits_write_image(image_, gsub(".hdf5", glue("_{filename_}.fits"), cat_name), keyvalues = hdr)
                          }

                          # Save segmentation images if requested.
                          if (img_config[["keepsegims"]]) {
                            if (single) {
                              save_image(profound_run$segim, "sci")
                            } else {
                              segims <- profound_run$segimlist
                              names <- profound_run$multibands
                              Map(save_image, segims, names)
                            }
                          }

                          # Close the hdf5 catalogue.
                          .self$close_groups(catalogue, photometry, "photometry")
                          .self$close_groups(catalogue, groups, "groups")
                          H5Fclose(catalogue)
                          h5closeAll()

                          return(cat_name)
                        }
                        ))