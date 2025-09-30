fmt:
	poetry run black ./*.py ./alg/ ./analysis ./envs ./scripts ./tests ./utils
	poetry run isort ./*.py ./alg/ ./analysis ./envs ./scripts ./tests ./utils

lint:
	poetry run black --check ./*.py ./alg/ ./analysis ./envs ./scripts ./tests ./utils
	poetry run isort --check ./*.py ./alg/ ./analysis ./envs ./scripts ./tests ./utils
	poetry run mypy ./*.py ./alg/ ./analysis ./envs ./scripts ./tests ./utils
	poetry run pflake8 ./*.py ./alg/ ./analysis ./envs ./scripts ./tests ./utils
	@echo "🎉 All checks passed successfully! ✨🍰✨"

test:
	poetry run pytest