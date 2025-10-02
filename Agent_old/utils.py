import re

class SQLCodeParser:
    """A utility class to parse and extract SQL code from LLM responses."""
    
    @staticmethod
    def extract_sql_code(response_text: str) -> str:
        """
        Extract clean SQL code from LLM response text.
        
        Args:
            response_text: The full response text from the LLM containing SQL
            
        Returns:
            Clean SQL code without markdown formatting or explanatory text
        """
        try:
            # Extract from ```sql code blocks
            sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
            
            if sql_blocks:
                sql_code = '\n\n'.join(sql_blocks)
            else:
                #Extract from ``` code blocks
                code_blocks = re.findall(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                sql_code = '\n\n'.join(code_blocks) if code_blocks else response_text
            

            cleaned_sql = SQLCodeParser._clean_sql_code(sql_code)
            
            return cleaned_sql
            
        except Exception as e:
            return f"-- Error parsing SQL: {str(e)}\n{response_text}"
    
    @staticmethod
    def _clean_sql_code(sql_code: str) -> str:
        """
        Clean and format the extracted SQL code.
        
        Args:
            sql_code: Raw SQL code string
            
        Returns:
            Cleaned and formatted SQL code
        """        
        
        # Remove comment lines that are not SQL comments
        lines = sql_code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped and not cleaned_lines:
                continue
                
            # keep SQL comments
            if stripped.startswith('--') or stripped.startswith('/*') or '*/' in stripped:
                cleaned_lines.append(line)
                continue

            if not stripped:
                continue
                
            # Check if line starts with SQL keywords
            if any(stripped.upper().startswith(keyword) for keyword in 
                    ['CREATE', 'INSERT', 'UPDATE', 'DELETE', 'SELECT', 'ALTER', 'DROP']):
                cleaned_lines.append(line)
                
            # Keep lines that are part of multi-line SQL statements
            elif stripped and (stripped.endswith(',') or stripped.endswith('(') or 
                             stripped.startswith(')') or 'VALUES' in stripped.upper() or
                             any(keyword in stripped.upper() for keyword in 
                                 ['PRIMARY KEY', 'FOREIGN KEY', 'NOT NULL', 'DEFAULT', 'CHECK'])):
                cleaned_lines.append(line)

            # Keep statement terminators and closing brackets
            elif stripped in [');', ')', ';'] or stripped.endswith(');'):
                cleaned_lines.append(line)
                
            # Keep lines that look like column definitions or table content
            elif '(' in stripped or ')' in stripped or ',' in stripped:
                cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines)
        
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        result = result.strip()
        
        return result
    
    @staticmethod
    def separate_statements(sql_code: str) -> dict:
        """
        Separate CREATE TABLE and INSERT statements.
        
        Args:
            sql_code: Complete SQL code string
            
        Returns:
            Dictionary with 'create_table' and 'insert_statements' keys
        """
        try:
            statements = []
            current_statement = ""
            in_string = False
            quote_char = None
            
            for char in sql_code:
                if char in ['"', "'"] and not in_string:
                    in_string = True
                    quote_char = char
                elif char == quote_char and in_string:
                    in_string = False
                    quote_char = None
                elif char == ';' and not in_string:
                    if current_statement.strip():
                        statements.append(current_statement.strip())
                    current_statement = ""
                    continue
                
                current_statement += char
            
            if current_statement.strip():
                statements.append(current_statement.strip())
            
            # Categorize statements
            create_tables = []
            insert_statements = []
            
            for statement in statements:
                statement_upper = statement.strip().upper()
                if statement_upper.startswith('CREATE TABLE'):
                    create_tables.append(statement)
                elif statement_upper.startswith('INSERT INTO'):
                    insert_statements.append(statement)
            
            return {
                'create_table': '\n\n'.join(create_tables),
                'insert_statements': '\n\n'.join(insert_statements),
                'all_statements': statements
            }
            
        except Exception as e:
            return {
                'create_table': f"-- Error separating statements: {str(e)}",
                'insert_statements': "",
                'all_statements': [sql_code]
            }