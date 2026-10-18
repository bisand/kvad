# kvad-relay's image: the static binary on distroless, which brings the CA
# roots the ACME client needs and nothing else (no shell, no package
# manager). Built from a binary cross-compiled on the host, so building the
# image runs nothing; see deploy/README.md.
FROM gcr.io/distroless/static-debian12:nonroot
COPY kvad-relay /kvad-relay
ENTRYPOINT ["/kvad-relay"]
