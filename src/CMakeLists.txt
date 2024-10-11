# SPDX-License-Identifier: Unlicense

target_sources( ${PROJECT_NAME}
    PRIVATE
        
    llama_context.cpp 
    llama_context.h 
    llama_model_loader.cpp 
    llama_model_loader.h 
    llama_model.cpp 
    llama_model.h 
    register_types.cpp 
    register_types.h
)

target_include_directories( ${PROJECT_NAME}
    PRIVATE
        "src"
)
