#!/bin/bash

echo 'starting spring boot'

cd code

echo 'installing dependencies'

./mvnw install

echo 'start programm'

./mvnw spring-boot:run


