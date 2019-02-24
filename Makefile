autopep8:
	autopep8 --ignore E501,E241,W690 --in-place --recursive --aggressive correlationMatrix/

lint:
	flake8 correlationMatrix

autolint: autopep8 lint

