
### Start:

### Command Line

#### Unix:
./mvnw clean ; ./mvnw install ; ./mvnw spring-boot:run

#### Windows:
./mvnw clean ; ./mvnw install ; ./mvnw spring-boot:run


### Docker with full Cuda Setup

#### requirements:

docker compose installed

#### Start

cd src/docker
docker build -t cuda-java ./
docker-compose up

Note: may require sudo, some installations require docker compose instead of docker-compose