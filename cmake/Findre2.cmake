find_path(re2_INCLUDE_DIR
  NAMES re2/re2.h
  PATHS /usr/local/include /usr/include
)

find_library(re2_LIBRARY
  NAMES re2 libre2
  PATHS /usr/local/lib /usr/local/lib64 /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(re2
  REQUIRED_VARS re2_INCLUDE_DIR re2_LIBRARY
)

if(re2_FOUND AND NOT TARGET re2::re2)
  add_library(re2::re2 UNKNOWN IMPORTED)
  set_target_properties(re2::re2 PROPERTIES
    IMPORTED_LOCATION "${re2_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${re2_INCLUDE_DIR}"
  )
endif()
