from typing import List, Dict
from app.models.schemas import LLMServiceRequestDto
from app.services.template import TemplateService
from app.core.message_utils import extract_text_from_content, normalize_role


class OpenAIMessageConverter:
    """Converts templates to OpenAI message format"""
    
    def __init__(self):
        self.template_service = TemplateService()
    
    def convert(self, request: LLMServiceRequestDto) -> List[Dict[str, str]]:
        """Convert request template to OpenAI messages"""
        filled_template = self._fill_template(request)
        return self._convert_messages(filled_template)
    
    def _fill_template(self, request: LLMServiceRequestDto):
        """Replace template placeholders with actual input values"""
        return self.template_service.render_template(
            template=request.prompt_version.template,
            template_format=request.prompt_version.template_format,
            inputs=request.inputs
        )
    
    def _convert_chat_messages(self, chat_template) -> List[Dict[str, str]]:
        """Convert chat template messages to OpenAI format"""
        messages = []
        
        for msg in chat_template.messages:
            content = extract_text_from_content(msg.content)
            role = normalize_role(msg.role)
            
            messages.append({
                "role": role,
                "content": content
            })
        
        return messages
    
    def _convert_string_template(self, string_template) -> List[Dict[str, str]]:
        """Convert string template to a single user message"""
        return [{
            "role": "user",
            "content": string_template.template
        }]
    
    def _convert_messages(self, filled_template) -> List[Dict[str, str]]:
        """Convert filled template (with values replaced) to OpenAI message format"""
        if filled_template.type == "chat":
            return self._convert_chat_messages(filled_template)
        elif filled_template.type == "string":
            return self._convert_string_template(filled_template)
        else:
            return []

