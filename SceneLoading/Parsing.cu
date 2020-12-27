#include "Parsing.h"
#include "../Utils/MathsCommons.h"
#include "../Core/Material.h"
#include "../Core/Primitive.h"
#include "../Samplers/SimpleSampler.h"
#include "../Integrators/DirectLightingIntegrator.h"
#include "../Integrators/PathTracingIntegrator.h"
#include "../Integrators/CreateIntegrator.h"
#include "../Utils/Utils.h"
#include <unordered_map>
#include <string>
#include "Lexing.h"
#include "../Core/Texture.h"
#include "../Core/Impl.h"

#define SIGNAL_PARSING_ERROR(err,pos,tokenString) SIGNAL_ERROR((std::string("Parsing Error: ")+err+std::string("\n at token ")+std::to_string(pos)+": "+tokenString).c_str())

template<typename T>
struct NamedStorage {
	std::unordered_map<std::string, T> items;
	bool has(const std::string& name) {
		return items.find(name) != items.end();
	}
	void add(const std::string& name, const T& item) {
		items.insert({ name, item });
	}
	T get(const std::string& name) {
		if (!has(name)) {
			SIGNAL_ERROR((std::string("Named T not found :")+name).c_str());
		}
		return items.at(name);
	}
};
using MaterialStorage = NamedStorage<MaterialObject>;
using TextureStorage = NamedStorage<Texture2D>;


std::vector<float> readNumList(TokenBuf& buf){
	std::vector<float> result;
	buf.checkAndPop<LeftSquareBracketToken>();
	while(buf.peek()->type == TokenType::Num){
		std::shared_ptr<NumToken> num = buf.checkAndPop<NumToken>();
		result.push_back(num->value);
		
	}
	buf.checkAndPop<RightSquareBracketToken>();
	return result;
}

std::vector<std::string> readStringList(TokenBuf& buf) {
	std::vector<std::string> result;
	buf.checkAndPop<LeftSquareBracketToken>();
	while (buf.peek()->type == TokenType::String) {
		std::shared_ptr<StringToken> s = buf.checkAndPop<StringToken>();
		result.push_back(s->all);

	}
	buf.checkAndPop<RightSquareBracketToken>();
	return result;
}

void readUntilNextKeyWorkd(TokenBuf& buf) {
	buf.checkAndPop<KeyWordToken>();
	while (buf.peek()->type != TokenType::KeyWord) {
		buf.moveForward();
	}
}

ObjectDefinition readObjectDefinition(TokenBuf& buf){
	ObjectDefinition def;

	std::shared_ptr<KeyWordToken> keyWord = buf.checkAndPop<KeyWordToken>();
	def.keyWord = keyWord -> word;

	std::shared_ptr<StringToken> name = buf.checkAndPop<StringToken>();
	def.objectName = name->words[0];

	while(buf.peek()->type == TokenType::String){
		std::shared_ptr<StringToken> key = buf.checkAndPop<StringToken>();
		std::string fieldName = key->words[key->words.size()-1];

		auto nextToken = buf.peek();
		switch(nextToken->type){
			case TokenType::String:
				def.params.strings[fieldName] = buf.checkAndPop<StringToken>()->all;
				break;
			case TokenType::Num:
				def.params.nums[fieldName] = buf.checkAndPop<NumToken>()->value;
				break;
			case TokenType::LeftSquareBracket:
				if (buf.peek(1)->type == TokenType::Num) {
					def.params.numLists[fieldName] = readNumList(buf);
				}
				else if (buf.peek(1)->type == TokenType::String) {
					def.params.stringLists[fieldName] = readStringList(buf);
				}
				else {
					SIGNAL_ERROR((std::string("Unaccepted List Element. Token index: ") + std::to_string(buf.currentIndex+1) + "." + buf.peek(1)->print()).c_str());
				}
				break;
			default:
				SIGNAL_ERROR((std::string("Read Object failed. Token index: ")+std::to_string(buf.currentIndex)+ "." + nextToken->print()).c_str());
				break;
		}
	}
	def.isDefined = true;
	return def;
}



void readLookAt(TokenBuf& buf, float3& eye, float3& center, float3& up){
	auto lookAt = buf.checkAndPop<KeyWordToken>();
	if(lookAt->word != "LookAt"){
		SIGNAL_ERROR("LookAt not found when calling readLookAt.");
	}
	eye.x = buf.checkAndPop<NumToken>()->value;
	eye.y = buf.checkAndPop<NumToken>()->value;
	eye.z = buf.checkAndPop<NumToken>()->value;

	center.x = buf.checkAndPop<NumToken>()->value;
	center.y = buf.checkAndPop<NumToken>()->value;
	center.z = buf.checkAndPop<NumToken>()->value;

	up.x = buf.checkAndPop<NumToken>()->value;
	up.y = buf.checkAndPop<NumToken>()->value;
	up.z = buf.checkAndPop<NumToken>()->value;
}


bool readTransform(TokenBuf& buf, glm::mat4& transform){
	auto nextToken = buf.peek();
	auto keyWord = std::dynamic_pointer_cast<KeyWordToken>(nextToken);
	if(keyWord){
		std::string word = keyWord->word;
		if(word=="Translate"){
			buf.moveForward();
			float x = buf.checkAndPop<NumToken>()->value;
			float y = buf.checkAndPop<NumToken>()->value;
			float z = buf.checkAndPop<NumToken>()->value;
			transform = glm::translate(transform,glm::vec3(x,y,z));
			return true;
		}
		if(word=="Rotate"){
			buf.moveForward();
			float angle = buf.checkAndPop<NumToken>()->value;
			float x = buf.checkAndPop<NumToken>()->value;
			float y = buf.checkAndPop<NumToken>()->value;
			float z = buf.checkAndPop<NumToken>()->value;
			transform = glm::rotate(transform,glm::radians(angle),glm::vec3(x,y,z));
			return true;
		}
		if(word=="Scale"){
			buf.moveForward();
			float x = buf.checkAndPop<NumToken>()->value;
			float y = buf.checkAndPop<NumToken>()->value;
			float z = buf.checkAndPop<NumToken>()->value;
			transform = glm::scale(transform,glm::vec3(x,y,z));
			return true;
		}
		if (word == "Transform") {
			buf.moveForward();
			std::vector<float> matData = readNumList(buf);
			glm::mat4 mat = to_mat4(matData);
			transform = mat * transform;
			return true;
		}

		else{
			return false;
		}
	}
	SIGNAL_PARSING_ERROR("Keyword expected.",buf.currentIndex,nextToken->print());
}


void parseSceneWideOptions(TokenBuf& buf,RenderSetup& result){

	glm::mat4 transform(1.0);

	ObjectDefinition cameraDef;
	ObjectDefinition filmDef;
	ObjectDefinition integratorDef;
	ObjectDefinition samplerDef;

	// parse scene-wide options
	while(true){
		auto nextToken = buf.peek();
		auto keyWord = std::dynamic_pointer_cast<KeyWordToken>(nextToken);
		if(keyWord){
			if(keyWord->word == "WorldBegin"){
				break;
			}
			else if(keyWord->word == "LookAt"){
				float3 eye,center,up;
				
				readLookAt(buf,eye,center,up);
				eye = apply(transform, eye);
				center = apply(transform, center);
				up = glm::mat3(transform) * up;
				transform = glm::lookAtLH(to_vec3(eye), to_vec3(center), to_vec3(up)) * transform;
			}
			else if(keyWord->word == "Camera"){
				cameraDef = readObjectDefinition(buf);
			}
			else if(keyWord->word == "Film"){
				filmDef = readObjectDefinition(buf);
			}
			else if(keyWord->word == "Sampler"){
				samplerDef = readObjectDefinition(buf);
			}
			else if (readTransform(buf, transform)) {

			}
			else if(keyWord->word == "Integrator"){
				integratorDef = readObjectDefinition(buf);
			}
			
			else{
				std::cout<<"reading unrecognized object from "<<buf.currentIndex;
				readUntilNextKeyWorkd(buf);
				std::cout<<"done"<<std::endl;
			}
		}
		else{
			SIGNAL_PARSING_ERROR("Keyword expected.",buf.currentIndex,nextToken->print());
		}
	}

	if(!(cameraDef.isDefined && filmDef.isDefined && integratorDef.isDefined && samplerDef.isDefined)){
		SIGNAL_ERROR("incomplete scene-wide options");
	}

	auto integrator = CreateIntegrator::createFromObjectDefinition(integratorDef);
	integrator->sampler = std::make_unique<SamplerObject>(SamplerObject::createFromObjectDefinition(samplerDef));

	result.renderer.integrator = std::move(integrator);
	result.renderer.film = std::make_unique<FilmObject>(FilmObject::createFromObjectDefinition(filmDef));
	int width = result.renderer.film->getWidth();
	int height = result.renderer.film->getHeight();
	result.renderer.camera = std::make_unique<CameraObject>(CameraObject::createFromObjectDefinition(cameraDef,glm::inverse(transform),width,height));

	if (filmDef.params.hasString("filename")) {
		result.outputFileName = filmDef.params.getString("filename");
	}
	else {
		result.outputFileName = "output.png";
	}

}



void parseSubsection(TokenBuf& buf, RenderSetup& result, glm::mat4 transform,const std::filesystem::path& basePath, MaterialStorage& materialsStore,TextureStorage& textureStore, bool& nextShapeHasAreaLight,std::shared_ptr<MaterialObject> currentMaterial) {
	auto begin = buf.checkAndPop<KeyWordToken>();
	if ( !endsWith(begin->word,"Begin") ) {
		SIGNAL_PARSING_ERROR("XXXBegin expected.", buf.currentIndex, begin->print());
	}

	std::string subsectionName = begin->word.substr(0, begin->word.size() - std::string("Begin").size());

	
	bool shouldReverseOrientation = false;

	while (true) {
		auto nextToken = buf.peek();
		auto keyWord = std::dynamic_pointer_cast<KeyWordToken>(nextToken);
		if (keyWord) {
			if (endsWith(keyWord->word, "End")) {
				std::string endingSubsectionName = keyWord->word.substr(0, keyWord->word.size() - std::string("End").size());
				if (endingSubsectionName != subsectionName) {
					SIGNAL_PARSING_ERROR("Mismatching subsection names.", buf.currentIndex, nextToken->print());
				}
				break;
			}
			else if (endsWith(keyWord->word,"Begin")) {
				parseSubsection(buf, result, transform, basePath,materialsStore,textureStore,nextShapeHasAreaLight,currentMaterial);
			}
			else if (keyWord->word == "Shape") {
				auto shapeDef = readObjectDefinition(buf);
				TriangleMesh shape = TriangleMesh::createFromObjectDefinition(shapeDef, transform, basePath);
				if (shouldReverseOrientation) {
					shape.reverseOrientation = true;
				}
				
				Primitive prim;
				
				prim.shape = shape;

				if (currentMaterial.get() != nullptr) {
					prim.material = *currentMaterial;
				}
				else {
					MaterialObject  matteGray = MatteMaterial(make_float3(1, 1, 1));
					prim.material = matteGray;
				}

				if (nextShapeHasAreaLight) {
					prim.setAreaLightIndex(result.scene.lightsHost.size() - 1);
					nextShapeHasAreaLight = false;
				}

				result.scene.primitivesHost.push_back(prim);
			}
			else if (keyWord->word == "AreaLightSource") {
				auto lightDef = readObjectDefinition(buf);
				LightObject light = LightObject::createFromObjectDefinition(lightDef, transform);
				if (light.is<DiffuseAreaLight>()) {
					DiffuseAreaLight* diffuseLight = light.get<DiffuseAreaLight>();
					// shapeIndex will be the index of the next shape to be added.
					diffuseLight->shapeIndex = result.scene.primitivesHost.size();
					std::cout << "added shape for diffuse area light "<< diffuseLight->shapeIndex << std::endl;
					nextShapeHasAreaLight = true;
				}

				result.scene.lightsHost.push_back(light);

			}
			else if (keyWord->word == "LightSource") {
				auto lightDef = readObjectDefinition(buf);
				LightObject light = LightObject::createFromObjectDefinition(lightDef, transform);
				result.scene.lightsHost.push_back(light);
				if (lightDef.objectName == "infinite") {
					result.scene.environmentMapIndex = result.scene.lightsHost.size() - 1;
				}
			}
			else if (readTransform(buf, transform)) {

			}
			else if (keyWord->word == "MakeNamedMaterial") {
				auto def = readObjectDefinition(buf);
				materialsStore.add(def.objectName,MaterialObject::createFromObjectDefinition(def,textureStore.items));
			}
			else if (keyWord->word == "Material") {
				auto def = readObjectDefinition(buf);
				def.params.strings["type"] = def.objectName;
				currentMaterial = std::make_shared<MaterialObject>(MaterialObject::createFromObjectDefinition(def, textureStore.items));
			}
			else if (keyWord->word == "Texture") {
				auto def = readObjectDefinition(buf);
				textureStore.add(def.objectName,Texture2D::createFromObjectDefinition(def,transform,basePath));
			}
			else if (keyWord->word == "NamedMaterial") {
				buf.moveForward();
				std::string name = buf.checkAndPop<StringToken>()->all;
				currentMaterial = std::make_shared<MaterialObject>(materialsStore.get(name));
			}
			else if (keyWord->word == "Include") {
				buf.moveForward();
				std::string includeFile = buf.checkAndPop<StringToken>()->all;
				std::filesystem::path includePath = basePath / std::filesystem::path(includeFile);
				TokenBuf includedTokens = runLexing(includePath);
				buf.insertHere(includedTokens);
			}
			else if (keyWord->word == "ReverseOrientation") {
				buf.moveForward();
				shouldReverseOrientation = true;
			}
			else {
				std::cout << "reading unrecognized object:" << keyWord->word<< " from " << buf.currentIndex ;
				readUntilNextKeyWorkd(buf);
				std::cout << "done" << std::endl;
			}
		}
		else {
			SIGNAL_PARSING_ERROR("Keyword expected.", buf.currentIndex, nextToken->print());
		}
	}

	buf.checkAndPop<KeyWordToken>();
}


RenderSetup runParsing(TokenBuf tokens, const std::filesystem::path& basePath) {

	RenderSetup result;
	MaterialStorage materials;
	TextureStorage textures;


	parseSceneWideOptions(tokens, result);

	bool nextShapeHasAreaLight = false;
	std::shared_ptr<MaterialObject> currentMaterial = nullptr;
	glm::mat4 currentTransform(1.0);

	parseSubsection(tokens,result,currentTransform,basePath,materials,textures,nextShapeHasAreaLight,currentMaterial);

	return result;
}