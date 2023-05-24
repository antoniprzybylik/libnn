#include <json/json.h>
#include <json/value.h>
#include <iostream>
#include <fstream>
#include <locale>

#include "simple_net.h"

const std::wstring classes[] = {
	L"zero",
	L"jeden",
	L"dwa",
	L"trzy",
	L"cztery",
	L"pięć",
	L"sześć",
	L"siedem",
	L"osiem",
	L"dziewięć",
};

int main(const int argc, const char *const argv[])
{
	SimpleNet nn;
	std::ifstream ifs;

	Json::Reader reader;
	Json::Value root;

	size_t i, j;
	ColumnVector<rl_t> input(in_size);
	ColumnVector<rl_t> output(out_size);

	std::locale::global(std::locale(""));
	std::wcout.imbue(std::locale());

	if (argc < 3) {
		std::wcerr << "This program accepts two parameters:\n"
			  << "* Network parameters file\n"
			  << "* Image to classify\n";
		return(0);
	}

	if (argc > 3) {
		std::wcerr << "Too many arguments.\n";
		return(0);
	}

	ifs.open(argv[1]);
	if (!reader.parse(ifs, root)) {
		std::wcerr << "Corrupted file.\n";
		return(-1);
	}

	nn.load(root);
	ifs.close();

	ifs.open(argv[2]);
	for (ColumnVector<rl_t>::iterator it = input.begin();
	     it != input.end() && ifs >> *it; it++);
	
	output = nn.forward(input);
	std::wcout << "KLASA           PRAWDOPODOBIEŃSTWO\n";
	for (i = 0; i < output.length(); i++) {
		std::wcout << classes[i];
		for (j = 0; j < 24 - classes[i].size(); j++)
			std::wcout << ' ';
		std::wcout << output[i] << '\n';
	}
	std::wcout << std::flush;
	
	return(0);
}
