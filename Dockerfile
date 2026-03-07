FROM alpine:3.23

WORKDIR /app

# Run crond in foreground with logging
CMD ["crond", "-f", "-l", "8"]
