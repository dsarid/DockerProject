FROM mongo:5

COPY . /script_mount

RUN chmod +x /script_mount/relset.sh

CMD ["/script_mount/relset.sh"]
