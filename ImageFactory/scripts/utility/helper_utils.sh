#!/bin/bash




format_bytes() {
    local bytes=$1
    if (( bytes < 1024 )); then
        echo "${bytes}B"
    elif (( bytes < 1048576 )); then
        echo "$((bytes / 1024))KB"
    elif (( bytes < 1073741824 )); then
        echo "$((bytes / 1048576))MB"
    elif (( bytes < 1099511627776 )); then
        echo "$((bytes / 1073741824))GB"
    else
        echo "$((bytes / 1099511627776))TB"
    fi
}