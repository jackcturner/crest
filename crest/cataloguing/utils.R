library(Rfits)

check_overlap <- function(x, y, positions, radius) {

  # Check if a given circular aperture falls within the area of another.

  # Arguments
  # ---------
  # x (int)
  #   x location of the new aperture.
  # y (int)
  #   y location of the new aperture.
  # positions (matrix)
  #   Nx2 matrix containing centres of placed apertures.
  # radius (float)
  #   The radius of the apertures.

  # Returns
  # -------
  # (bool)
  #   TRUE if no overlap with any aperture, FALSE otherwise.

  for (i in seq_len(nrow(positions))) {
    if (!is.na(positions[i, 1])) {

      # Calculate the distance between the new and old aperture.
      dx <- positions[i, 1] - x
      dy <- positions[i, 2] - y
      distance <- sqrt(dx^2 + dy^2)

      # Apertures overlap.
      if (distance < 2 * radius) {
        return(FALSE)
      }
    }
  }

  # No overlap.
  return(TRUE)
}

centroid_com <- function(image) {

  # Calculate the centroid of an image using moments.

  # Arguments
  # ---------
  # image (matrix)
  #   The 2D image from which to measure the centroid.

  # Returns
  # -------
  # centre (list)
  #   List containg the x-y coordinate of the source centre.

  x <- seq_len(ncol(image))
  y <- seq_len(nrow(image))

  # Get a grid of x and y values.
  x_grid <- rep(x, each = nrow(image))
  y_grid <- rep(y, times = ncol(image))
  flux <- as.vector(image)

  # Get the centre by weighting by flux.
  x_centre <- sum(x_grid * flux) / sum(flux)
  y_centre <- sum(y_grid * flux) / sum(flux)

  centre <- c(x_centre, y_centre)

  return(centre)
}

aperture_photometry <- function(image, position, radius, subsample = 5) {

  # Perform circular aperture photometry at a single location.

  # Arguments
  # ---------
  # image (matrix)
  #   The 2D image from which to measure photometry.
  # position (list)
  #   The x-y position of the aperture to place.
  # radius (float)
  #   The radius of the circular aperture to use.
  # subsample (int)
  #   The degree of pixel subsampling.

  # Returns
  # -------
  # total_flux (float)
  #   The flux measured in the aperture.

  # Place the aperture here.
  x_centre <- position[1]
  y_centre <- position[2]

  # Will sum up the flux as we go.
  total_flux <- 0

  # Loop over all pixels in a region around the aperture.
  for (i in floor(x_centre - radius - 1):ceiling(x_centre + radius + 1)) {
    for (j in floor(y_centre - radius - 1):ceiling(y_centre + radius + 1)) {
      if (i >= 1 && i <= ncol(image) && j >= 1 && j <= nrow(image)) {

        # Subpixel grid within this pixel.
        sub_x <- seq(i - 0.5, i + 0.5, length.out = subsample)
        sub_y <- seq(j - 0.5, j + 0.5, length.out = subsample)

        # Create a grid of subsample points.
        sub_grid <- expand.grid(x = sub_x, y = sub_y)
        distances <- sqrt((sub_grid$x - x_centre)^2 + (sub_grid$y - y_centre)^2)

        # Fraction of subpixels inside aperture.
        fraction_inside <- mean(distances <= radius)

        # Add fractional flux contribution.
        total_flux <- total_flux + fraction_inside * image[j, i]
      }
    }
  }

  return(total_flux)
}

measure_curve_of_growth <- function(image, radii, position = NULL, norm = TRUE, show = FALSE) {

  # Measure the Curve of Growth and profile of a source.

  # Arguments
  # ---------
  # image (matrix)
  #   The 2D image from which to measure the COG.
  # radii (list)
  #   The radii steps at which to measure the COG.
  # position (list)
  #   The x-y position of the source centre. If NULL, compute internally
  #   from moments.
  # norm (bool)
  #   Should the COG/profile be returned normalised by the maximum value?
  # show (bool)
  #   Should the COG be plotted?

  # Returns
  # -------
  # radii (list)
  #   The steps at which the COG was measured. Same as input.
  # cog (list)
  #   The measured COG.
  # profile (list)
  #   The measured profile.

  # Measure the centroid of the image if required.
  if (is.null(position)) {
    position <- centroid_com(image)
  }

  # Measure the COG using circular apertures.
  cog <- sapply(radii, function(r) aperture_photometry(image, position, r, subsample = 5))

  # Get the area enclosed by each aperture.
  area <- pi * radii^2

  # Calculate the area differences.
  area_cog <- c(area[1], diff(area))

  # Calculate the profile.
  profile <- c(cog[1], diff(cog)) / area_cog

  # Normalise.
  if (norm) {
    cog <- cog / max(cog)
    profile <- profile / max(profile)
  }

  return(list(radii = radii, cog = cog, profile = profile))
}