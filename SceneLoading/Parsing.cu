#include "Parsing.h"
#include "../Utils/MathsCommons.h"
#include "../Shapes/ShapeObject.h"
#include "../Core/Material.h"
#include "../Core/Primitive.h"
#include "../Samplers/NaiveCameraSampler.h"
#include "../Integrators/DirectLightingGPUIntegrator.h"

#define SIGNAL_PARSING_ERROR(err,pos,tokenString) SIGNAL_ERROR((std::string("Parsing Error: ")+err+std::string("\n at token ")+std::to_string(pos)+": "+tokenString).c_str())





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


bool doRotation(TokenBuf& buf, glm::mat4& transform){
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
		else{
			return false;
		}
	}
	SIGNAL_PARSING_ERROR("Keyword expected.",buf.currentIndex,nextToken->print());
}


void parseSceneWideOptions(TokenBuf& buf,RenderSetup& result){
	float3 eye,center,up;
	bool hasLookAt = false;
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
				readLookAt(buf,eye,center,up);
				hasLookAt = true;
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
			else if(keyWord->word == "Integrator"){
				integratorDef = readObjectDefinition(buf);
			}
			else{
				std::cout<<"reading unrecognized object from "<<buf.currentIndex<<std::endl;
				readObjectDefinition(buf);
				std::cout<<"done"<<std::endl;
			}
		}
		else{
			SIGNAL_PARSING_ERROR("Keyword expected.",buf.currentIndex,nextToken->print());
		}
	}

	if(!(hasLookAt && cameraDef.isDefined && filmDef.isDefined && integratorDef.isDefined && samplerDef.isDefined)){
		SIGNAL_ERROR("incomplete scene-wide options");
	}

	auto integrator = std::make_unique<DirectLightingGPUIntegrator>();
	integrator->cameraSampler = std::make_unique<NaiveCameraSampler>();

	result.renderer.integrator = std::move(integrator);
	result.renderer.film = std::make_unique<FilmObject>(FilmObject::createFromObjectDefinition(filmDef));
	int width = result.renderer.film->getWidth();
	int height = result.renderer.film->getHeight();
	result.renderer.camera = std::make_unique<CameraObject>(CameraObject::createFromObjectDefinition(cameraDef,eye,center,up,width,height));

}





void readAttribute(TokenBuf& buf,RenderSetup& result,glm::mat4 transform, const std::filesystem::path& basePath){
	auto begin = buf.checkAndPop<KeyWordToken>();
	if(begin->word != "AttributeBegin"){
		SIGNAL_PARSING_ERROR("AttributeBegin expected.",buf.currentIndex,begin->print());
	}


	while(true){
		auto nextToken = buf.peek();
		auto keyWord = std::dynamic_pointer_cast<KeyWordToken>(nextToken);
		if(keyWord){
			if(keyWord->word == "AttributeEnd"){
				break;
			}
			else if(keyWord->word == "Shape"){
				auto shapeDef = readObjectDefinition(buf);
				ShapeObject shape = ShapeObject::createFromObjectDefinition(shapeDef,transform,basePath);
				Primitive prim;
				Material lambertian;
    			lambertian.bsdf = LambertianBSDF(make_float3(1, 1, 1));
				prim.shape = shape;
				prim.material = lambertian;
				result.scene.primitivesHost.push_back(prim);
			}
			else if(keyWord->word == "LightSource"){
				auto lightDef = readObjectDefinition(buf);
				LightObject light = LightObject::createFromObjectDefinition(lightDef,transform);
				result.scene.lightsHost.push_back(light);
			}
			else if(doRotation(buf,transform)){

			}
			else{
				std::cout<<"reading unrecognized object from "<<buf.currentIndex<<std::endl;
				readObjectDefinition(buf);
				std::cout<<"done"<<std::endl;
			}
		}
		else{
			SIGNAL_PARSING_ERROR("Keyword expected.",buf.currentIndex,nextToken->print());
		}
	}
	buf.checkAndPop<KeyWordToken>();
}


void parseWorld(TokenBuf& buf,RenderSetup& result, const std::filesystem::path& basePath){
	auto worldBegin = buf.checkAndPop<KeyWordToken>();
	if(worldBegin->word != "WorldBegin"){
		SIGNAL_PARSING_ERROR("WorldBegin expected.",buf.currentIndex,worldBegin->print());
	}

	glm::mat4 transform(1.0);

	while(true){
		auto nextToken = buf.peek();
		auto keyWord = std::dynamic_pointer_cast<KeyWordToken>(nextToken);
		if(keyWord){
			if(keyWord->word == "WorldEnd"){
				break;
			}
			else if(keyWord->word == "AttributeBegin"){
				readAttribute(buf, result,transform,basePath);
			}
			else if(keyWord->word == "Shape"){
				auto shapeDef = readObjectDefinition(buf);
				ShapeObject shape = ShapeObject::createFromObjectDefinition(shapeDef,transform,basePath);
				Primitive prim;
				Material lambertian;
    			lambertian.bsdf = LambertianBSDF(make_float3(1, 1, 1));
				prim.shape = shape;
				prim.material = lambertian;
				result.scene.primitivesHost.push_back(prim);
			}
			else if(keyWord->word == "LightSource"){
				auto lightDef = readObjectDefinition(buf);
				LightObject light = LightObject::createFromObjectDefinition(lightDef,transform);
				result.scene.lightsHost.push_back(light);
			}
			else if(doRotation(buf,transform)){
				
			}
			else{
				std::cout<<"reading unrecognized object from "<<buf.currentIndex<<std::endl;
				readObjectDefinition(buf);
				std::cout<<"done"<<std::endl;
			}
		}
		else{
			SIGNAL_PARSING_ERROR("Keyword expected.",buf.currentIndex,nextToken->print());
		}
	}
	result.scene.environmentMapIndex = 0;
	buf.checkAndPop<KeyWordToken>();
}


RenderSetup runParsing(TokenBuf tokens, const std::filesystem::path& basePath) {

	RenderSetup result;

	parseSceneWideOptions(tokens, result);

	parseWorld(tokens,result,basePath);

	return result;
}